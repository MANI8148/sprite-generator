import json
import os
import shutil
from typing import Optional


class FileStorage:
    def __init__(self, base_dir: str = "data/storage"):
        self.base_dir = base_dir
        self._history_path = os.path.join(base_dir, "history.json")
        self._ensure_dirs()

    def _ensure_dirs(self):
        os.makedirs(self.base_dir, exist_ok=True)

    def _load_history(self) -> list:
        if not os.path.isfile(self._history_path):
            return []
        with open(self._history_path, "r") as f:
            return json.load(f)

    def _save_history(self, history: list):
        with open(self._history_path, "w") as f:
            json.dump(history, f, indent=2)

    def add_job(self, job_id: str, entry: dict):
        history = self._load_history()
        entry["job_id"] = job_id
        history.append(entry)
        self._save_history(history)

    def get_job(self, job_id: str) -> Optional[dict]:
        for entry in self._load_history():
            if entry["job_id"] == job_id:
                return entry
        return None

    def list_jobs(self) -> list:
        return self._load_history()

    def get_output_dir(self, job_id: str) -> str:
        return os.path.join(self.base_dir, job_id)

    def ensure_output_dir(self, job_id: str) -> str:
        d = self.get_output_dir(job_id)
        os.makedirs(d, exist_ok=True)
        return d

    def clear(self):
        if os.path.isdir(self.base_dir):
            shutil.rmtree(self.base_dir)
        self._ensure_dirs()
