"""Tests for structured logging and correlation IDs (Phase 5 item: Proper error handling, structured logging, correlation IDs)."""

import json
import time
import tempfile
import os
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.api.routes import set_pipeline, set_generator_loaded, set_storage, set_library, _batch_jobs
from backend.modules.pipeline.orchestrator import AssetPipeline
from backend.modules.storage.file_storage import FileStorage
from backend.modules.storage.asset_library import AssetLibrary
from backend.modules.rate_limiter import RateLimiter, set_rate_limiter
from backend.modules.tasks.queue import TaskQueue, set_task_queue
from backend.modules.logging.correlation import (
    generate_correlation_id,
    get_correlation_id,
    set_correlation_id,
    _correlation_id,
)
from backend.modules.logging.structured_logger import StructuredLogger, get_logger, _loggers


class FakeGenerator:
    def __init__(self):
        self.loaded = False

    def load(self):
        self.loaded = True

    def unload(self):
        self.loaded = False

    def generate(self, prompt="", negative_prompt="", width=512, height=512, seed=-1, num_images=None):
        from PIL import Image
        import numpy as np
        n = num_images or 1
        images = []
        for _ in range(n):
            arr = np.zeros((64, 64, 4), dtype=np.uint8)
            arr[:, :, :3] = [255, 0, 0]
            arr[:, :, 3] = 255
            images.append(Image.fromarray(arr, "RGBA"))
        return images


@pytest.fixture(autouse=True)
def reset_state():
    set_generator_loaded(False)
    tmp = tempfile.mkdtemp()
    set_storage(FileStorage(base_dir=tmp))
    set_library(AssetLibrary(base_dir=os.path.join(tmp, "lib")))
    set_rate_limiter(RateLimiter(max_requests=100, window_seconds=60))
    set_task_queue(TaskQueue(max_workers=4))
    _batch_jobs.clear()


@pytest.fixture
def client():
    pipe = AssetPipeline()
    pipe.set_generator(FakeGenerator())
    set_pipeline(pipe)
    set_generator_loaded(True)
    return TestClient(app)


class TestCorrelationID:
    def test_correlation_id_generated(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200
        assert "X-Correlation-ID" in resp.headers
        corr_id = resp.headers["X-Correlation-ID"]
        assert len(corr_id) == 12
        assert corr_id.isalnum()

    def test_correlation_id_echoed(self, client):
        resp = client.get("/health", headers={"X-Correlation-ID": "my-test-id"})
        assert resp.status_code == 200
        assert resp.headers["X-Correlation-ID"] == "my-test-id"

    def test_different_requests_different_ids(self, client):
        resp1 = client.get("/health")
        resp2 = client.get("/health")
        assert resp1.headers["X-Correlation-ID"] != resp2.headers["X-Correlation-ID"]

    def test_correlation_id_on_generate(self, client):
        resp = client.post("/generate", json={"asset_type": "character"})
        assert resp.status_code == 202
        assert "X-Correlation-ID" in resp.headers


class TestCorrelationIDUnit:
    def test_generate_id(self):
        cid = generate_correlation_id()
        assert len(cid) == 12
        assert isinstance(cid, str)

    def test_set_and_get(self):
        set_correlation_id("test-456")
        assert get_correlation_id() == "test-456"

    def test_default_empty(self):
        _correlation_id.set("")
        assert _correlation_id.get() == ""


@pytest.fixture(autouse=True)
def reset_loggers():
    _loggers.clear()
    _correlation_id.set("")


class TestStructuredLogger:
    def test_logger_outputs_json(self, capsys):
        logger = StructuredLogger("test_logger")
        logger.info("test_event", key="value", count=42)
        captured = capsys.readouterr().out
        record = json.loads(captured)
        assert record["event"] == "test_event"
        assert record["level"] == "info"
        assert record["logger"] == "test_logger"
        assert record["data"]["key"] == "value"
        assert record["data"]["count"] == 42
        assert "timestamp" in record
        assert "correlation_id" in record

    def test_logger_error_level(self, capsys):
        logger = StructuredLogger("error_logger")
        logger.error("error_event", error_msg="something broke")
        captured = capsys.readouterr().out
        record = json.loads(captured)
        assert record["level"] == "error"
        assert record["event"] == "error_event"

    def test_logger_warning_level(self, capsys):
        logger = StructuredLogger("warn_logger")
        logger.warning("warn_event")
        captured = capsys.readouterr().out
        record = json.loads(captured)
        assert record["level"] == "warning"

    def test_logger_debug_level(self, capsys):
        logger = StructuredLogger("debug_logger", level=10)
        logger.debug("debug_event", detail="test")
        captured = capsys.readouterr().out
        record = json.loads(captured)
        assert record["level"] == "debug"

    def test_logger_no_extra_data(self, capsys):
        logger = StructuredLogger("no_data_logger")
        logger.info("simple_event")
        captured = capsys.readouterr().out
        record = json.loads(captured)
        assert "data" not in record
        assert record["event"] == "simple_event"

    def test_logger_correlation_id_included(self, capsys):
        set_correlation_id("corr-42")
        logger = StructuredLogger("corr_logger")
        logger.info("check_corr")
        captured = capsys.readouterr().out
        record = json.loads(captured)
        assert record["correlation_id"] == "corr-42"

    def test_get_logger_singleton(self):
        logger1 = get_logger("singleton_test")
        logger2 = get_logger("singleton_test")
        assert logger1 is logger2

    def test_get_logger_different_names(self):
        logger1 = get_logger("name_a")
        logger2 = get_logger("name_b")
        assert logger1 is not logger2


class TestRequestLogging:
    def test_health_request_logged(self, client, caplog):
        caplog.set_level(10)
        resp = client.get("/health")
        assert resp.status_code == 200
        request_records = [r for r in caplog.records if r.msg and "request" in str(r.msg)]
        assert len(request_records) >= 1

    def test_generate_post_logged(self, client, caplog):
        caplog.set_level(10)
        resp = client.post("/generate", json={"asset_type": "character"})
        assert resp.status_code == 202
        request_records = [r for r in caplog.records if r.msg and "request" in str(r.msg)]
        assert len(request_records) >= 1


class TestRateLimitLogging:
    def test_rate_limit_exceeded_logged(self, client, caplog):
        caplog.set_level(10)
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        set_rate_limiter(limiter)

        resp1 = client.post("/generate", json={"asset_type": "character"})
        assert resp1.status_code == 202

        resp2 = client.post("/generate", json={"asset_type": "enemy"})
        assert resp2.status_code == 429

        warn_records = [r for r in caplog.records if r.msg and "rate_limit_exceeded" in str(r.msg)]
        assert len(warn_records) >= 1
