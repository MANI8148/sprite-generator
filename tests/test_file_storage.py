"""Tests for file-based storage (roadmap: Phase 1 Item 3)."""

import os
import json
import tempfile

import pytest

from backend.modules.storage.file_storage import FileStorage


@pytest.fixture
def storage():
    tmp = tempfile.mkdtemp()
    return FileStorage(base_dir=tmp)


class TestFileStorageInit:
    def test_creates_base_dir(self):
        tmp = tempfile.mkdtemp()
        base = os.path.join(tmp, "sprite_storage")
        fs = FileStorage(base_dir=base)
        assert os.path.isdir(base)

    def test_uses_provided_base_dir(self):
        tmp = tempfile.mkdtemp()
        fs = FileStorage(base_dir=tmp)
        assert fs.base_dir == tmp


class TestFileStorageAddJob:
    def test_add_job_writes_to_history(self, storage):
        storage.add_job("job001", {"prompt": "a character"})
        history = storage.list_jobs()
        assert len(history) == 1
        assert history[0]["job_id"] == "job001"
        assert history[0]["prompt"] == "a character"

    def test_add_job_persists_to_disk(self, storage):
        storage.add_job("job002", {"prompt": "an enemy"})
        history_path = os.path.join(storage.base_dir, "history.json")
        assert os.path.isfile(history_path)
        with open(history_path) as f:
            data = json.load(f)
        assert len(data) == 1
        assert data[0]["job_id"] == "job002"

    def test_add_multiple_jobs(self, storage):
        storage.add_job("j1", {"prompt": "p1"})
        storage.add_job("j2", {"prompt": "p2"})
        storage.add_job("j3", {"prompt": "p3"})
        assert len(storage.list_jobs()) == 3


class TestFileStorageGetJob:
    def test_get_existing_job(self, storage):
        storage.add_job("abc", {"prompt": "test"})
        entry = storage.get_job("abc")
        assert entry is not None
        assert entry["prompt"] == "test"

    def test_get_nonexistent_job(self, storage):
        entry = storage.get_job("nonexistent")
        assert entry is None

    def test_get_job_after_reload(self, storage):
        storage.add_job("persist", {"prompt": "survives"})
        fs2 = FileStorage(base_dir=storage.base_dir)
        entry = fs2.get_job("persist")
        assert entry is not None


class TestFileStorageListJobs:
    def test_empty_history(self, storage):
        assert storage.list_jobs() == []

    def test_list_returns_all(self, storage):
        storage.add_job("a", {"prompt": "x"})
        storage.add_job("b", {"prompt": "y"})
        jobs = storage.list_jobs()
        assert len(jobs) == 2

    def test_list_returns_copies(self, storage):
        storage.add_job("x", {"prompt": "data"})
        jobs = storage.list_jobs()
        jobs.append({"fake": "entry"})
        assert len(storage.list_jobs()) == 1


class TestFileStorageOutputDir:
    def test_get_output_dir(self, storage):
        d = storage.get_output_dir("myjob")
        assert d.endswith("myjob")

    def test_ensure_output_dir_creates(self, storage):
        d = storage.ensure_output_dir("newjob")
        assert os.path.isdir(d)

    def test_ensure_output_dir_is_idempotent(self, storage):
        d1 = storage.ensure_output_dir("samejob")
        d2 = storage.ensure_output_dir("samejob")
        assert d1 == d2
        assert os.path.isdir(d1)


class TestFileStorageClear:
    def test_clear_removes_all_data(self, storage):
        storage.add_job("j1", {"prompt": "p1"})
        storage.add_job("j2", {"prompt": "p2"})
        storage.clear()
        assert storage.list_jobs() == []

    def test_clear_returns_empty_list(self, storage):
        storage.add_job("j1", {"prompt": "p1"})
        storage.clear()
        assert storage.list_jobs() == []

    def test_clear_base_dir_exists(self, storage):
        storage.ensure_output_dir("somejob")
        storage.clear()
        assert os.path.isdir(storage.base_dir)
