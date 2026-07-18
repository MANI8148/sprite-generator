import json
import logging
import os
from typing import Optional, List

logger = logging.getLogger(__name__)


class R2Storage:
    def __init__(
        self,
        bucket: Optional[str] = None,
        endpoint: Optional[str] = None,
        access_key_id: Optional[str] = None,
        secret_access_key: Optional[str] = None,
        public_url: Optional[str] = None,
    ):
        self._bucket_name = bucket or os.environ.get("R2_BUCKET", "")
        self._endpoint = endpoint or os.environ.get("R2_ENDPOINT", "")
        self._access_key_id = access_key_id or os.environ.get("R2_ACCESS_KEY_ID", "")
        self._secret_access_key = secret_access_key or os.environ.get("R2_SECRET_ACCESS_KEY", "")
        self._public_url = public_url or os.environ.get("R2_PUBLIC_URL", "")
        self._history_key = "history/history.json"
        self._client = None
        self._bucket = None
        self._available = False

        if self._bucket_name and self._endpoint and self._access_key_id and self._secret_access_key:
            self._init_client()

    def _init_client(self):
        try:
            import boto3
            self._client = boto3.client(
                "s3",
                endpoint_url=self._endpoint,
                aws_access_key_id=self._access_key_id,
                aws_secret_access_key=self._secret_access_key,
            )
            try:
                self._client.head_bucket(Bucket=self._bucket_name)
            except Exception:
                self._client.create_bucket(Bucket=self._bucket_name)
            self._available = True
        except ImportError:
            logger.warning("boto3 not installed; R2Storage unavailable")
        except Exception as e:
            logger.warning("R2Storage unavailable: %s", e)

    @property
    def available(self) -> bool:
        return self._available

    def _ensure_available(self):
        if not self._available:
            raise RuntimeError("R2Storage not configured. Set R2_ENDPOINT, R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, and R2_BUCKET environment variables.")

    def _load_history(self) -> list:
        if not self._available:
            return []
        try:
            response = self._client.get_object(Bucket=self._bucket_name, Key=self._history_key)
            return json.loads(response["Body"].read().decode("utf-8"))
        except Exception:
            return []

    def _save_history(self, history: list):
        self._ensure_available()
        self._client.put_object(
            Bucket=self._bucket_name,
            Key=self._history_key,
            Body=json.dumps(history, indent=2).encode("utf-8"),
            ContentType="application/json",
        )

    def add_job(self, job_id: str, entry: dict):
        if not self._available:
            logger.warning("R2Storage not available; skipping add_job for %s", job_id)
            return
        history = self._load_history()
        entry["job_id"] = job_id
        history.append(entry)
        self._save_history(history)

    def get_job(self, job_id: str) -> Optional[dict]:
        if not self._available:
            return None
        for entry in self._load_history():
            if entry["job_id"] == job_id:
                return entry
        return None

    def list_jobs(self) -> list:
        if not self._available:
            return []
        return self._load_history()

    def get_output_dir(self, job_id: str) -> str:
        return os.path.join("data", "storage", job_id)

    def ensure_output_dir(self, job_id: str) -> str:
        d = self.get_output_dir(job_id)
        os.makedirs(d, exist_ok=True)
        return d

    def upload_file(self, local_path: str, object_key: str) -> str:
        self._ensure_available()
        self._client.upload_file(Filename=local_path, Bucket=self._bucket_name, Key=object_key)
        if self._public_url:
            return f"{self._public_url.rstrip('/')}/{object_key}"
        return f"{self._endpoint.rstrip('/')}/{self._bucket_name}/{object_key}"

    def download_file(self, object_key: str, local_path: str):
        self._ensure_available()
        os.makedirs(os.path.dirname(local_path) or ".", exist_ok=True)
        self._client.download_file(Bucket=self._bucket_name, Key=object_key, Filename=local_path)

    def list_objects(self, prefix: str = "") -> List[str]:
        self._ensure_available()
        keys = []
        paginator = self._client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=self._bucket_name, Prefix=prefix):
            for obj in page.get("Contents", []):
                keys.append(obj["Key"])
        return keys

    def delete_object(self, object_key: str):
        self._ensure_available()
        self._client.delete_object(Bucket=self._bucket_name, Key=object_key)

    def clear(self):
        if not self._available:
            return
        try:
            keys = self.list_objects()
            if keys:
                self._client.delete_objects(
                    Bucket=self._bucket_name,
                    Delete={"Objects": [{"Key": k} for k in keys]},
                )
        except Exception as e:
            logger.warning("R2Storage clear failed: %s", e)
