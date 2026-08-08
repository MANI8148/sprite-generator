"""Tests for global error handling (Phase 5 roadmap item: proper error
handling, structured logging, correlation IDs).

Covers the ``register_error_handlers`` wiring: unhandled exceptions return
structured JSON 500s and are logged with the request's correlation ID,
``HTTPException``/unknown-route errors keep their status + detail, and
``RequestValidationError`` produces a structured 422. Every error response
carries the ``X-Correlation-ID`` header.
"""

import logging

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from backend.modules.logging.correlation import generate_correlation_id, set_correlation_id
from backend.modules.logging.error_handlers import register_error_handlers


@pytest.fixture
def client():
    app = FastAPI()

    @app.middleware("http")
    async def correlation_middleware(request, call_next):
        corr_id = request.headers.get("X-Correlation-ID") or generate_correlation_id()
        set_correlation_id(corr_id)
        response = await call_next(request)
        response.headers["X-Correlation-ID"] = corr_id
        return response

    register_error_handlers(app)

    @app.get("/boom")
    def boom():
        raise ValueError("kaboom")

    @app.get("/http-error")
    def http_error():
        raise HTTPException(status_code=404, detail="not here")

    @app.get("/http-server-error")
    def http_server_error():
        raise HTTPException(status_code=503, detail="service unavailable")

    @app.get("/items/{item_id}")
    def item(item_id: int):
        return {"item_id": item_id}

    return TestClient(app, raise_server_exceptions=False)


class TestUnhandledExceptions:
    def test_returns_json_500(self, client):
        resp = client.get("/boom")
        assert resp.status_code == 500
        assert resp.headers["content-type"].startswith("application/json")
        assert resp.json() == {"detail": "Internal server error"}

    def test_correlation_id_header_present(self, client):
        resp = client.get("/boom")
        assert resp.headers.get("X-Correlation-ID")

    def test_request_correlation_id_echoed(self, client):
        resp = client.get("/boom", headers={"X-Correlation-ID": "trace-me"})
        assert resp.headers["X-Correlation-ID"] == "trace-me"

    def test_unhandled_exception_logged_structured(self, client, caplog):
        caplog.set_level(logging.DEBUG)
        client.get("/boom")
        records = [
            r for r in caplog.records
            if r.msg and "unhandled_exception" in str(r.msg)
        ]
        assert records, "expected a structured unhandled_exception log record"
        import json as _json

        record = _json.loads(records[0].msg)
        assert record["level"] == "error"
        assert record["event"] == "unhandled_exception"
        assert record["data"]["exception_type"] == "ValueError"
        assert "ValueError: kaboom" in record["data"]["traceback"]
        assert record["correlation_id"]


class TestHTTPExceptions:
    def test_client_error_keeps_status_and_detail(self, client):
        resp = client.get("/http-error")
        assert resp.status_code == 404
        assert resp.json() == {"detail": "not here"}
        assert resp.headers.get("X-Correlation-ID")

    def test_server_error_keeps_status_and_detail(self, client):
        resp = client.get("/http-server-error")
        assert resp.status_code == 503
        assert resp.json() == {"detail": "service unavailable"}
        assert resp.headers.get("X-Correlation-ID")


class TestRequestValidationErrors:
    def test_returns_422_with_field_errors(self, client):
        resp = client.get("/items/not-an-int")
        assert resp.status_code == 422
        body = resp.json()
        assert isinstance(body["detail"], list)
        assert any("item_id" in str(e.get("loc")) for e in body["detail"])
        assert resp.headers.get("X-Correlation-ID")

    def test_validation_error_logged(self, client, caplog):
        caplog.set_level(logging.DEBUG)
        client.get("/items/not-an-int")
        records = [
            r for r in caplog.records
            if r.msg and "validation_error" in str(r.msg)
        ]
        assert records


class TestIntegration:
    def test_unknown_route_returns_json_404(self):
        from backend.main import app

        client = TestClient(app)
        resp = client.get("/this/route/does/not/exist")
        assert resp.status_code == 404
        assert resp.json() == {"detail": "Not Found"}
        assert resp.headers.get("X-Correlation-ID")

    def test_health_still_works_with_handlers_registered(self):
        from backend.main import app

        client = TestClient(app)
        resp = client.get("/health")
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"
