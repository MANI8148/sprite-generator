import json
import os
import shutil
import threading
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional


@dataclass
class AssetRecord:
    asset_id: str
    job_id: str
    asset_type: str
    prompt: str
    quality_tier: str
    tags: list = field(default_factory=list)
    category: str = ""
    thumbnail_path: Optional[str] = None
    zip_path: Optional[str] = None
    output_paths: list = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""
    metadata: dict = field(default_factory=dict)
    generation_hash: str = ""

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.utcnow().isoformat() + "Z"
        if not self.updated_at:
            self.updated_at = self.created_at


class AssetLibrary:
    def __init__(self, base_dir: str = "data/library"):
        self.base_dir = base_dir
        self._index_path = os.path.join(base_dir, "library_index.json")
        self._lock = threading.Lock()
        self._ensure_dirs()

    def _ensure_dirs(self):
        with self._lock:
            os.makedirs(self.base_dir, exist_ok=True)

    def _load_index(self) -> dict:
        if not os.path.isfile(self._index_path):
            return {}
        with open(self._index_path, "r") as f:
            return json.load(f)

    def _save_index(self, index: dict):
        with open(self._index_path, "w") as f:
            json.dump(index, f, indent=2)

    def _read_index(self):
        with self._lock:
            return self._load_index()

    def _write_index(self, index: dict):
        with self._lock:
            self._save_index(index)

    def _record_to_dict(self, record: AssetRecord) -> dict:
        return asdict(record)

    def _dict_to_record(self, data: dict) -> AssetRecord:
        return AssetRecord(**data)

    def add_asset(self, record: AssetRecord) -> str:
        if not record.asset_id:
            record.asset_id = str(uuid.uuid4())[:8]
        record.updated_at = datetime.utcnow().isoformat() + "Z"
        with self._lock:
            index = self._load_index()
            index[record.asset_id] = self._record_to_dict(record)
            self._save_index(index)
        return record.asset_id

    def get_asset(self, asset_id: str) -> Optional[AssetRecord]:
        with self._lock:
            index = self._load_index()
            data = index.get(asset_id)
        if data is None:
            return None
        return self._dict_to_record(data)

    def update_asset(self, asset_id: str, **updates) -> Optional[AssetRecord]:
        with self._lock:
            index = self._load_index()
            if asset_id not in index:
                return None
            record = index[asset_id]
            for key, value in updates.items():
                if key in ("asset_id", "created_at"):
                    continue
                record[key] = value
            record["updated_at"] = datetime.utcnow().isoformat() + "Z"
            index[asset_id] = record
            self._save_index(index)
            return self._dict_to_record(record)

    def delete_asset(self, asset_id: str) -> bool:
        with self._lock:
            index = self._load_index()
            if asset_id not in index:
                return False
            del index[asset_id]
            self._save_index(index)
            return True

    def list_assets(
        self,
        asset_type: Optional[str] = None,
        quality_tier: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[list] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list:
        with self._lock:
            index = self._load_index()
        records = [self._dict_to_record(d) for d in index.values()]

        if asset_type:
            records = [r for r in records if r.asset_type == asset_type]
        if quality_tier:
            records = [r for r in records if r.quality_tier == quality_tier]
        if category:
            records = [r for r in records if r.category == category]
        if tags:
            records = [r for r in records if any(t in r.tags for t in tags)]
        if search:
            q = search.lower()
            records = [
                r for r in records
                if q in r.prompt.lower() or q in r.asset_type.lower() or q in r.asset_id
            ]

        records.sort(key=lambda r: r.created_at, reverse=True)
        return records[offset:offset + limit]

    def list_tags(self) -> list:
        with self._lock:
            index = self._load_index()
        tags = set()
        for data in index.values():
            tags.update(data.get("tags", []))
        return sorted(tags)

    def get_asset_dir(self, asset_id: str) -> str:
        return os.path.join(self.base_dir, asset_id)

    def ensure_asset_dir(self, asset_id: str) -> str:
        d = self.get_asset_dir(asset_id)
        os.makedirs(d, exist_ok=True)
        return d

    def add_tags(self, asset_id: str, tags: list) -> Optional[AssetRecord]:
        with self._lock:
            index = self._load_index()
            if asset_id not in index:
                return None
            existing = set(index[asset_id].get("tags", []))
            existing.update(tags)
            index[asset_id]["tags"] = sorted(existing)
            index[asset_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"
            self._save_index(index)
            return self._dict_to_record(index[asset_id])

    def remove_tags(self, asset_id: str, tags: list) -> Optional[AssetRecord]:
        with self._lock:
            index = self._load_index()
            if asset_id not in index:
                return None
            existing = set(index[asset_id].get("tags", []))
            for t in tags:
                existing.discard(t)
            index[asset_id]["tags"] = sorted(existing)
            index[asset_id]["updated_at"] = datetime.utcnow().isoformat() + "Z"
            self._save_index(index)
            return self._dict_to_record(index[asset_id])

    def find_by_generation_hash(self, generation_hash: str) -> Optional[AssetRecord]:
        with self._lock:
            index = self._load_index()
        for data in index.values():
            if data.get("metadata", {}).get("generation_hash") == generation_hash:
                return self._dict_to_record(data)
            if data.get("generation_hash") == generation_hash:
                return self._dict_to_record(data)
        return None

    def count(self) -> int:
        with self._lock:
            return len(self._load_index())

    def clear(self):
        with self._lock:
            if os.path.isdir(self.base_dir):
                shutil.rmtree(self.base_dir)
            self._ensure_dirs()
