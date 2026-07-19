import json
import os
import sqlite3
import threading
from datetime import datetime
from typing import Optional, List

from .asset_library import AssetRecord


class DatabaseLibrary:
    def __init__(self, db_path: str = "data/library.db"):
        self.db_path = db_path
        self._lock = threading.Lock()
        self._init_db()

    def _init_db(self):
        with self._lock:
            os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS assets (
                        asset_id TEXT PRIMARY KEY,
                        job_id TEXT NOT NULL,
                        asset_type TEXT NOT NULL,
                        prompt TEXT NOT NULL,
                        quality_tier TEXT NOT NULL,
                        tags TEXT DEFAULT '[]',
                        category TEXT DEFAULT '',
                        thumbnail_path TEXT,
                        zip_path TEXT,
                        output_paths TEXT DEFAULT '[]',
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        metadata TEXT DEFAULT '{}',
                        generation_hash TEXT DEFAULT ''
                    )
                """)
                conn.execute("""
                    CREATE TABLE IF NOT EXISTS tags (
                        tag TEXT PRIMARY KEY
                    )
                """)
                try:
                    conn.execute("ALTER TABLE assets ADD COLUMN generation_hash TEXT DEFAULT ''")
                except sqlite3.OperationalError:
                    pass
                conn.commit()
            finally:
                conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> AssetRecord:
        return AssetRecord(
            asset_id=row["asset_id"],
            job_id=row["job_id"],
            asset_type=row["asset_type"],
            prompt=row["prompt"],
            quality_tier=row["quality_tier"],
            tags=json.loads(row["tags"]),
            category=row["category"],
            thumbnail_path=row["thumbnail_path"],
            zip_path=row["zip_path"],
            output_paths=json.loads(row["output_paths"]),
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            metadata=json.loads(row["metadata"]),
            generation_hash=row["generation_hash"] if "generation_hash" in row.keys() else "",
        )

    def add_asset(self, record: AssetRecord) -> str:
        now = datetime.utcnow().isoformat() + "Z"
        if not record.asset_id:
            import uuid
            record.asset_id = str(uuid.uuid4())[:8]
        if not record.created_at:
            record.created_at = now
        record.updated_at = now

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute(
                    """INSERT OR REPLACE INTO assets
                       (asset_id, job_id, asset_type, prompt, quality_tier,
                        tags, category, thumbnail_path, zip_path, output_paths,
                        created_at, updated_at, metadata, generation_hash)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        record.asset_id, record.job_id, record.asset_type,
                        record.prompt, record.quality_tier,
                        json.dumps(record.tags), record.category,
                        record.thumbnail_path, record.zip_path,
                        json.dumps(record.output_paths),
                        record.created_at, record.updated_at,
                        json.dumps(record.metadata),
                        record.generation_hash,
                    ),
                )
                for tag in record.tags:
                    conn.execute("INSERT OR IGNORE INTO tags (tag) VALUES (?)", (tag,))
                conn.commit()
            finally:
                conn.close()
        return record.asset_id

    def get_asset(self, asset_id: str) -> Optional[AssetRecord]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
                )
                row = cursor.fetchone()
                return self._row_to_record(row) if row else None
            finally:
                conn.close()

    def update_asset(self, asset_id: str, **updates) -> Optional[AssetRecord]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return None
                record = self._row_to_record(row)
                for key, value in updates.items():
                    if key in ("asset_id", "created_at"):
                        continue
                    if key in ("tags", "output_paths", "metadata"):
                        setattr(record, key, value)
                        value = json.dumps(value)
                    setattr(record, key, value)
                record.updated_at = datetime.utcnow().isoformat() + "Z"
                conn.execute(
                    """UPDATE assets SET
                       job_id=?, asset_type=?, prompt=?, quality_tier=?,
                       tags=?, category=?, thumbnail_path=?, zip_path=?,
                       output_paths=?, updated_at=?, metadata=?
                       WHERE asset_id=?""",
                    (
                        record.job_id, record.asset_type, record.prompt,
                        record.quality_tier, json.dumps(record.tags),
                        record.category, record.thumbnail_path,
                        record.zip_path, json.dumps(record.output_paths),
                        record.updated_at, json.dumps(record.metadata),
                        record.asset_id,
                    ),
                )
                conn.commit()
                return record
            finally:
                conn.close()

    def delete_asset(self, asset_id: str) -> bool:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute(
                    "DELETE FROM assets WHERE asset_id = ?", (asset_id,)
                )
                conn.commit()
                return cursor.rowcount > 0
            finally:
                conn.close()

    def list_assets(
        self,
        asset_type: Optional[str] = None,
        quality_tier: Optional[str] = None,
        category: Optional[str] = None,
        tags: Optional[list] = None,
        search: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[AssetRecord]:
        query = "SELECT * FROM assets WHERE 1=1"
        params = []

        if asset_type:
            query += " AND asset_type = ?"
            params.append(asset_type)
        if quality_tier:
            query += " AND quality_tier = ?"
            params.append(quality_tier)
        if category:
            query += " AND category = ?"
            params.append(category)
        if tags:
            query += " AND (" + " OR ".join("tags LIKE ?" for _ in tags) + ")"
            params.extend(f'%{t}%' for t in tags)
        if search:
            q = f"%{search.lower()}%"
            query += " AND (LOWER(prompt) LIKE ? OR LOWER(asset_type) LIKE ? OR asset_id LIKE ?)"
            params.extend([q, q, q])

        query += " ORDER BY created_at DESC LIMIT ? OFFSET ?"
        params.extend([limit, offset])

        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(query, params)
                return [self._row_to_record(row) for row in cursor.fetchall()]
            finally:
                conn.close()

    def list_tags(self) -> list:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("SELECT tag FROM tags ORDER BY tag")
                return [row[0] for row in cursor.fetchall()]
            finally:
                conn.close()

    def count(self) -> int:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                cursor = conn.execute("SELECT COUNT(*) FROM assets")
                return cursor.fetchone()[0]
            finally:
                conn.close()

    def find_by_generation_hash(self, generation_hash: str) -> Optional[AssetRecord]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM assets WHERE generation_hash = ? LIMIT 1",
                    (generation_hash,),
                )
                row = cursor.fetchone()
                if row is None:
                    cursor = conn.execute(
                        "SELECT * FROM assets WHERE json_extract(metadata, '$.generation_hash') = ? LIMIT 1",
                        (generation_hash,),
                    )
                    row = cursor.fetchone()
                return self._row_to_record(row) if row else None
            finally:
                conn.close()

    def get_asset_dir(self, asset_id: str) -> str:
        return os.path.join(os.path.dirname(self.db_path), "library", asset_id)

    def ensure_asset_dir(self, asset_id: str) -> str:
        d = self.get_asset_dir(asset_id)
        os.makedirs(d, exist_ok=True)
        return d

    def add_tags(self, asset_id: str, tags: list) -> Optional[AssetRecord]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return None
                existing = set(json.loads(row["tags"]))
                existing.update(tags)
                new_tags = sorted(existing)
                conn.execute(
                    "UPDATE assets SET tags=?, updated_at=? WHERE asset_id=?",
                    (json.dumps(new_tags), datetime.utcnow().isoformat() + "Z", asset_id),
                )
                for tag in tags:
                    conn.execute("INSERT OR IGNORE INTO tags (tag) VALUES (?)", (tag,))
                conn.commit()
                cursor = conn.execute("SELECT * FROM assets WHERE asset_id = ?", (asset_id,))
                return self._row_to_record(cursor.fetchone())
            finally:
                conn.close()

    def remove_tags(self, asset_id: str, tags: list) -> Optional[AssetRecord]:
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.row_factory = sqlite3.Row
                cursor = conn.execute(
                    "SELECT * FROM assets WHERE asset_id = ?", (asset_id,)
                )
                row = cursor.fetchone()
                if not row:
                    return None
                existing = set(json.loads(row["tags"]))
                for t in tags:
                    existing.discard(t)
                new_tags = sorted(existing)
                conn.execute(
                    "UPDATE assets SET tags=?, updated_at=? WHERE asset_id=?",
                    (json.dumps(new_tags), datetime.utcnow().isoformat() + "Z", asset_id),
                )
                conn.commit()
                cursor = conn.execute("SELECT * FROM assets WHERE asset_id = ?", (asset_id,))
                return self._row_to_record(cursor.fetchone())
            finally:
                conn.close()

    def clear(self):
        with self._lock:
            conn = sqlite3.connect(self.db_path)
            try:
                conn.execute("DELETE FROM assets")
                conn.execute("DELETE FROM tags")
                conn.commit()
            finally:
                conn.close()
