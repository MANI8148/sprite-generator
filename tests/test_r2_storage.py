"""Tests for Cloudflare R2 cloud storage (roadmap: Phase 3 Item 1)."""

import os
import json
import tempfile
from unittest.mock import patch, MagicMock, PropertyMock

import pytest

from backend.modules.storage.r2_storage import R2Storage


@pytest.fixture
def mock_s3_client():
    with patch("boto3.client") as mock_client:
        client = MagicMock()
        mock_client.return_value = client
        yield client


@pytest.fixture
def r2_storage(mock_s3_client):
    return R2Storage(
        bucket="test-bucket",
        endpoint="https://test.r2.cloudflarestorage.com",
        access_key_id="test-key",
        secret_access_key="test-secret",
    )


class TestR2StorageInit:
    def test_creates_client_when_env_vars_set(self):
        with patch("boto3.client") as mock_client:
            client = MagicMock()
            mock_client.return_value = client
            storage = R2Storage(
                bucket="b",
                endpoint="https://e.com",
                access_key_id="k",
                secret_access_key="s",
            )
            assert storage.available is True
            mock_client.assert_called_once_with(
                "s3",
                endpoint_url="https://e.com",
                aws_access_key_id="k",
                aws_secret_access_key="s",
            )

    def test_creates_bucket_if_not_exists(self, mock_s3_client):
        mock_s3_client.head_bucket.side_effect = Exception("Not found")
        storage = R2Storage(
            bucket="new-bucket",
            endpoint="https://e.com",
            access_key_id="k",
            secret_access_key="s",
        )
        assert storage.available is True
        mock_s3_client.create_bucket.assert_called_once_with(Bucket="new-bucket")

    def test_not_available_without_credentials(self):
        storage = R2Storage()
        assert storage.available is False

    def test_not_available_when_boto3_missing(self):
        with patch.dict("sys.modules", {"boto3": None}):
            with patch("builtins.__import__", side_effect=ImportError("no boto3")):
                storage = R2Storage(
                    bucket="b",
                    endpoint="https://e.com",
                    access_key_id="k",
                    secret_access_key="s",
                )
                assert storage.available is False


class TestR2StorageAddJob:
    def test_add_job_saves_to_history(self, r2_storage, mock_s3_client):
        mock_s3_client.get_object.side_effect = Exception("No history yet")
        r2_storage.add_job("job001", {"prompt": "a character"})

        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"] == "history/history.json"
        body = json.loads(call_kwargs["Body"].decode("utf-8"))
        assert len(body) == 1
        assert body[0]["job_id"] == "job001"
        assert body[0]["prompt"] == "a character"

    def test_add_job_appends_to_existing_history(self, r2_storage, mock_s3_client):
        existing = [{"job_id": "old", "prompt": "old job"}]
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(existing).encode("utf-8"))
        }

        r2_storage.add_job("new", {"prompt": "new job"})

        call_kwargs = mock_s3_client.put_object.call_args[1]
        body = json.loads(call_kwargs["Body"].decode("utf-8"))
        assert len(body) == 2
        assert body[0]["job_id"] == "old"
        assert body[1]["job_id"] == "new"

    def test_add_multiple_jobs(self, r2_storage, mock_s3_client):
        mock_s3_client.get_object.side_effect = [
            Exception("empty"),
            Exception("empty"),
            Exception("empty"),
        ]
        r2_storage.add_job("j1", {"prompt": "p1"})
        r2_storage.add_job("j2", {"prompt": "p2"})

        assert mock_s3_client.put_object.call_count == 2

    def test_add_job_warns_when_not_available(self):
        storage = R2Storage()
        with patch("logging.Logger.warning") as mock_warn:
            storage.add_job("j1", {"prompt": "test"})
            mock_warn.assert_called_once()


class TestR2StorageGetJob:
    def test_get_existing_job(self, r2_storage, mock_s3_client):
        existing = [{"job_id": "abc", "prompt": "test"}]
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(existing).encode("utf-8"))
        }
        entry = r2_storage.get_job("abc")
        assert entry is not None
        assert entry["prompt"] == "test"

    def test_get_nonexistent_job(self, r2_storage, mock_s3_client):
        existing = [{"job_id": "abc", "prompt": "test"}]
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(existing).encode("utf-8"))
        }
        entry = r2_storage.get_job("nonexistent")
        assert entry is None

    def test_get_job_when_no_history(self, r2_storage, mock_s3_client):
        mock_s3_client.get_object.side_effect = Exception("Not found")
        entry = r2_storage.get_job("abc")
        assert entry is None

    def test_get_job_when_not_available(self):
        storage = R2Storage()
        assert storage.get_job("abc") is None


class TestR2StorageListJobs:
    def test_empty_history(self, r2_storage, mock_s3_client):
        mock_s3_client.get_object.side_effect = Exception("Not found")
        assert r2_storage.list_jobs() == []

    def test_list_returns_all(self, r2_storage, mock_s3_client):
        existing = [{"job_id": "a"}, {"job_id": "b"}]
        mock_s3_client.get_object.return_value = {
            "Body": MagicMock(read=lambda: json.dumps(existing).encode("utf-8"))
        }
        jobs = r2_storage.list_jobs()
        assert len(jobs) == 2

    def test_list_when_not_available(self):
        storage = R2Storage()
        assert storage.list_jobs() == []


class TestR2StorageFileOps:
    def test_upload_file(self, r2_storage, mock_s3_client):
        tmp = tempfile.mkdtemp()
        local_path = os.path.join(tmp, "test.png")
        with open(local_path, "w") as f:
            f.write("fake png")

        url = r2_storage.upload_file(local_path, "jobs/job1/test.png")
        mock_s3_client.upload_file.assert_called_once_with(
            Filename=local_path, Bucket="test-bucket", Key="jobs/job1/test.png"
        )
        assert "jobs/job1/test.png" in url

    def test_upload_file_with_public_url(self, mock_s3_client):
        storage = R2Storage(
            bucket="b",
            endpoint="https://e.com",
            access_key_id="k",
            secret_access_key="s",
            public_url="https://assets.example.com",
        )
        tmp = tempfile.mkdtemp()
        local_path = os.path.join(tmp, "test.png")
        with open(local_path, "w") as f:
            f.write("fake png")

        url = storage.upload_file(local_path, "jobs/j1/test.png")
        assert url == "https://assets.example.com/jobs/j1/test.png"

    def test_download_file(self, r2_storage, mock_s3_client):
        tmp = tempfile.mkdtemp()
        local_path = os.path.join(tmp, "download.png")
        r2_storage.download_file("jobs/j1/test.png", local_path)
        mock_s3_client.download_file.assert_called_once_with(
            Bucket="test-bucket", Key="jobs/j1/test.png", Filename=local_path
        )
        assert os.path.isdir(tmp)

    def test_list_objects(self, r2_storage, mock_s3_client):
        paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "a.png"}, {"Key": "b.png"}]}
        ]

        keys = r2_storage.list_objects(prefix="jobs/")
        assert keys == ["a.png", "b.png"]
        paginator.paginate.assert_called_once_with(
            Bucket="test-bucket", Prefix="jobs/"
        )

    def test_delete_object(self, r2_storage, mock_s3_client):
        r2_storage.delete_object("jobs/j1/test.png")
        mock_s3_client.delete_object.assert_called_once_with(
            Bucket="test-bucket", Key="jobs/j1/test.png"
        )

    def test_upload_raises_when_not_available(self):
        storage = R2Storage()
        with pytest.raises(RuntimeError, match="R2Storage not configured"):
            storage.upload_file("/tmp/x.png", "key")

    def test_download_raises_when_not_available(self):
        storage = R2Storage()
        with pytest.raises(RuntimeError, match="R2Storage not configured"):
            storage.download_file("key", "/tmp/x.png")

    def test_list_objects_raises_when_not_available(self):
        storage = R2Storage()
        with pytest.raises(RuntimeError, match="R2Storage not configured"):
            storage.list_objects()

    def test_delete_object_raises_when_not_available(self):
        storage = R2Storage()
        with pytest.raises(RuntimeError, match="R2Storage not configured"):
            storage.delete_object("key")


class TestR2StorageClear:
    def test_clear_deletes_all_objects(self, r2_storage, mock_s3_client):
        paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [
            {"Contents": [{"Key": "a.png"}, {"Key": "b.png"}]}
        ]

        r2_storage.clear()

        mock_s3_client.delete_objects.assert_called_once_with(
            Bucket="test-bucket",
            Delete={"Objects": [{"Key": "a.png"}, {"Key": "b.png"}]},
        )

    def test_clear_empty_bucket(self, r2_storage, mock_s3_client):
        paginator = MagicMock()
        mock_s3_client.get_paginator.return_value = paginator
        paginator.paginate.return_value = [{"Contents": []}]

        r2_storage.clear()
        mock_s3_client.delete_objects.assert_not_called()

    def test_clear_when_not_available(self):
        storage = R2Storage()
        storage.clear()


class TestR2StorageOutputDir:
    def test_get_output_dir(self, r2_storage):
        d = r2_storage.get_output_dir("myjob")
        assert d.endswith("myjob")

    def test_ensure_output_dir_creates(self, r2_storage):
        tmp = tempfile.mkdtemp()
        with patch("backend.modules.storage.r2_storage.os.makedirs") as mock_mkdir:
            d = r2_storage.ensure_output_dir("newjob")
            mock_mkdir.assert_called_once()


class TestR2StorageEnvConfig:
    def test_configured_via_env_vars(self, mock_s3_client):
        with patch.dict(os.environ, {
            "R2_BUCKET": "env-bucket",
            "R2_ENDPOINT": "https://env.example.com",
            "R2_ACCESS_KEY_ID": "env-key",
            "R2_SECRET_ACCESS_KEY": "env-secret",
        }):
            storage = R2Storage()
            assert storage.available is True
            assert storage._bucket_name == "env-bucket"
            assert storage._endpoint == "https://env.example.com"
