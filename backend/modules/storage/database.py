import json
import os
import sqlite3
import threading
import contextlib
from datetime import datetime
from typing import Optional, List

from .asset_library import AssetRecord

try:
    import psycopg2
    from psycopg2.extras import RealDictCursor as _RealDictCursor

    _POSTGRES_AVAILABLE = True
except ImportError:  # pragma: no cover - psycopg2 is optional
    psycopg2 = None
    _RealDictCursor = None
    _POSTGRES_AVAILABLE = False

_ASSET_COLUMNS = [
    "asset_id", "job_id", "asset_type", "prompt", "quality_tier",
    "tags", "category", "thumbnail_path", "zip_path", "output_paths",
    "created_at", "updated_at", "metadata", "generation_hash",
]

_UPDATE_COLUMNS = [
    "job_id", "asset_type", "prompt", "quality_tier", "tags", "category",
    "thumbnail_path", "zip_path", "output_paths", "updated_at", "metadata",
]

_ASSETS_DDL = """
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
"""

_TAGS_DDL = """
    CREATE TABLE IF NOT EXISTS tags (
        tag TEXT PRIMARY KEY
    )
"""

_JOB_COLUMNS = [
    "job_id", "status", "prompt", "quality_tier", "validation",
    "zip_path", "output_paths", "error", "batch_id",
    "created_at", "updated_at",
]

_JOB_UPDATE_COLUMNS = [
    "status", "prompt", "quality_tier", "validation",
    "zip_path", "output_paths", "error", "batch_id", "updated_at",
]

_JOBS_DDL = """
    CREATE TABLE IF NOT EXISTS jobs (
        job_id TEXT PRIMARY KEY,
        status TEXT NOT NULL,
        prompt TEXT DEFAULT '',
        quality_tier TEXT DEFAULT '',
        validation TEXT DEFAULT '{}',
        zip_path TEXT,
        output_paths TEXT DEFAULT '[]',
        error TEXT DEFAULT '',
        batch_id TEXT DEFAULT '',
        created_at TEXT NOT NULL,
        updated_at TEXT NOT NULL
    )
"""


def _connect_postgres(database_url: str):
    if not _POSTGRES_AVAILABLE:
        raise RuntimeError(
            "PostgreSQL backend requires the 'psycopg2-binary' package. "
            "Install it with: pip install psycopg2-binary"
        )
    return psycopg2.connect(database_url, cursor_factory=_RealDictCursor)


def create_database_library(
    db_path: str = "data/library.db",
    database_url: Optional[str] = None,
) -> "DatabaseLibrary":
    """Create the asset library database.

    Defaults to SQLite. Set ``database_url`` (or the ``DATABASE_URL``
    environment variable) to a ``postgres://`` / ``postgresql://`` URL to use
    the PostgreSQL backend instead. SQLite remains the default so the
    SQLite -> PostgreSQL migration only kicks in when actually needed.
    """
    if not database_url:
        database_url = os.environ.get("DATABASE_URL")
    return DatabaseLibrary(db_path=db_path, database_url=database_url)


class DatabaseLibrary:
    """Persistent asset library backed by SQLite (default) or PostgreSQL.

    The two backends share the same interface and SQL shape; only the
    placeholders, upsert syntax, and a couple of JSON/lookup expressions
    differ. See ``create_database_library`` for how the backend is chosen.
    """

    def __init__(
        self,
        db_path: str = "data/library.db",
        database_url: Optional[str] = None,
    ):
        self.db_path = db_path
        if database_url is None:
            database_url = os.environ.get("DATABASE_URL")
        self.database_url = database_url
        self.backend = (
            "postgres"
            if database_url and database_url.startswith(("postgres://", "postgresql://"))
            else "sqlite"
        )
        self._lock = threading.Lock()
        self._init_db()

    @property
    def _ph(self) -> str:
        return "%s" if self.backend == "postgres" else "?"

    @contextlib.contextmanager
    def _connection(self):
        if self.backend == "sqlite":
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
        else:
            conn = _connect_postgres(self.database_url)
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _execute(self, conn, sql: str, params=()):
        cur = conn.cursor()
        cur.execute(sql, params)
        return cur

    def _init_db(self):
        with self._lock:
            if self.backend == "sqlite":
                os.makedirs(os.path.dirname(self.db_path) or ".", exist_ok=True)
            with self._connection() as conn:
                if self.backend == "sqlite":
                    self._execute(conn, "PRAGMA journal_mode=WAL")
                self._execute(conn, _ASSETS_DDL)
                self._execute(conn, _TAGS_DDL)
                self._execute(conn, _JOBS_DDL)
                self._ensure_generation_hash_column(conn)

    def _ensure_generation_hash_column(self, conn):
        if self.backend == "sqlite":
            try:
                self._execute(conn, "ALTER TABLE assets ADD COLUMN generation_hash TEXT DEFAULT ''")
            except sqlite3.OperationalError:
                pass
            return
        cur = self._execute(
            conn,
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'assets' AND column_name = 'generation_hash'",
        )
        if cur.fetchone() is None:
            self._execute(conn, "ALTER TABLE assets ADD COLUMN generation_hash TEXT DEFAULT ''")

    def _insert_asset_sql(self) -> str:
        cols = ", ".join(_ASSET_COLUMNS)
        ph = ", ".join([self._ph] * len(_ASSET_COLUMNS))
        if self.backend == "sqlite":
            return f"INSERT OR REPLACE INTO assets ({cols}) VALUES ({ph})"
        set_cols = [c for c in _ASSET_COLUMNS if c not in ("asset_id", "created_at")]
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in set_cols)
        return f"INSERT INTO assets ({cols}) VALUES ({ph}) ON CONFLICT (asset_id) DO UPDATE SET {set_clause}"

    def _insert_tag_sql(self) -> str:
        if self.backend == "sqlite":
            return f"INSERT OR IGNORE INTO tags (tag) VALUES ({self._ph})"
        return f"INSERT INTO tags (tag) VALUES ({self._ph}) ON CONFLICT (tag) DO NOTHING"

    def _update_asset_sql(self) -> str:
        set_clause = ", ".join(f"{c} = {self._ph}" for c in _UPDATE_COLUMNS)
        return f"UPDATE assets SET {set_clause} WHERE asset_id = {self._ph}"

    def _find_hash_sql(self) -> str:
        meta_expr = (
            "metadata->>'generation_hash'"
            if self.backend == "postgres"
            else "json_extract(metadata, '$.generation_hash')"
        )
        return f"SELECT * FROM assets WHERE {meta_expr} = {self._ph} LIMIT 1"

    def _row_to_record(self, row) -> AssetRecord:
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

        params = (
            record.asset_id, record.job_id, record.asset_type, record.prompt,
            record.quality_tier, json.dumps(record.tags), record.category,
            record.thumbnail_path, record.zip_path, json.dumps(record.output_paths),
            record.created_at, record.updated_at, json.dumps(record.metadata),
            record.generation_hash,
        )
        with self._connection() as conn:
            self._execute(conn, self._insert_asset_sql(), params)
            for tag in record.tags:
                self._execute(conn, self._insert_tag_sql(), (tag,))
        return record.asset_id

    def get_asset(self, asset_id: str) -> Optional[AssetRecord]:
        with self._connection() as conn:
            cur = self._execute(
                conn, f"SELECT * FROM assets WHERE asset_id = {self._ph}", (asset_id,)
            )
            row = cur.fetchone()
            return self._row_to_record(row) if row else None

    def update_asset(self, asset_id: str, **updates) -> Optional[AssetRecord]:
        with self._connection() as conn:
            cur = self._execute(
                conn, f"SELECT * FROM assets WHERE asset_id = {self._ph}", (asset_id,)
            )
            row = cur.fetchone()
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
            params = (
                record.job_id, record.asset_type, record.prompt, record.quality_tier,
                json.dumps(record.tags), record.category, record.thumbnail_path,
                record.zip_path, json.dumps(record.output_paths), record.updated_at,
                json.dumps(record.metadata), record.asset_id,
            )
            self._execute(conn, self._update_asset_sql(), params)
            return record

    def delete_asset(self, asset_id: str) -> bool:
        with self._connection() as conn:
            cur = self._execute(
                conn, f"DELETE FROM assets WHERE asset_id = {self._ph}", (asset_id,)
            )
            return cur.rowcount > 0

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
            query += f" AND asset_type = {self._ph}"
            params.append(asset_type)
        if quality_tier:
            query += f" AND quality_tier = {self._ph}"
            params.append(quality_tier)
        if category:
            query += f" AND category = {self._ph}"
            params.append(category)
        if tags:
            query += " AND (" + " OR ".join(f"tags LIKE {self._ph}" for _ in tags) + ")"
            params.extend(f'%{t}%' for t in tags)
        if search:
            q = f"%{search.lower()}%"
            query += (
                f" AND (LOWER(prompt) LIKE {self._ph} OR LOWER(asset_type) LIKE {self._ph} OR asset_id LIKE {self._ph})"
            )
            params.extend([q, q, q])

        query += f" ORDER BY created_at DESC LIMIT {self._ph} OFFSET {self._ph}"
        params.extend([limit, offset])

        with self._connection() as conn:
            cur = self._execute(conn, query, params)
            return [self._row_to_record(row) for row in cur.fetchall()]

    def list_tags(self) -> list:
        with self._connection() as conn:
            cur = self._execute(conn, "SELECT tag FROM tags ORDER BY tag")
            return [row["tag"] for row in cur.fetchall()]

    def count(self) -> int:
        with self._connection() as conn:
            cur = self._execute(conn, "SELECT COUNT(*) AS count FROM assets")
            return cur.fetchone()["count"]

    def find_by_generation_hash(self, generation_hash: str) -> Optional[AssetRecord]:
        with self._connection() as conn:
            cur = self._execute(
                conn, f"SELECT * FROM assets WHERE generation_hash = {self._ph} LIMIT 1",
                (generation_hash,),
            )
            row = cur.fetchone()
            if row is None:
                cur = self._execute(conn, self._find_hash_sql(), (generation_hash,))
                row = cur.fetchone()
            return self._row_to_record(row) if row else None

    def get_asset_dir(self, asset_id: str) -> str:
        return os.path.join(os.path.dirname(self.db_path), "library", asset_id)

    def ensure_asset_dir(self, asset_id: str) -> str:
        d = self.get_asset_dir(asset_id)
        os.makedirs(d, exist_ok=True)
        return d

    def add_tags(self, asset_id: str, tags: list) -> Optional[AssetRecord]:
        with self._connection() as conn:
            cur = self._execute(
                conn, f"SELECT * FROM assets WHERE asset_id = {self._ph}", (asset_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            existing = set(json.loads(row["tags"]))
            existing.update(tags)
            new_tags = sorted(existing)
            self._execute(
                conn,
                f"UPDATE assets SET tags = {self._ph}, updated_at = {self._ph} WHERE asset_id = {self._ph}",
                (json.dumps(new_tags), datetime.utcnow().isoformat() + "Z", asset_id),
            )
            for tag in tags:
                self._execute(conn, self._insert_tag_sql(), (tag,))
            cur = self._execute(
                conn, f"SELECT * FROM assets WHERE asset_id = {self._ph}", (asset_id,)
            )
            return self._row_to_record(cur.fetchone())

    def remove_tags(self, asset_id: str, tags: list) -> Optional[AssetRecord]:
        with self._connection() as conn:
            cur = self._execute(
                conn, f"SELECT * FROM assets WHERE asset_id = {self._ph}", (asset_id,)
            )
            row = cur.fetchone()
            if not row:
                return None
            existing = set(json.loads(row["tags"]))
            for t in tags:
                existing.discard(t)
            new_tags = sorted(existing)
            self._execute(
                conn,
                f"UPDATE assets SET tags = {self._ph}, updated_at = {self._ph} WHERE asset_id = {self._ph}",
                (json.dumps(new_tags), datetime.utcnow().isoformat() + "Z", asset_id),
            )
            cur = self._execute(
                conn, f"SELECT * FROM assets WHERE asset_id = {self._ph}", (asset_id,)
            )
            return self._row_to_record(cur.fetchone())

    def _insert_job_sql(self) -> str:
        cols = ", ".join(_JOB_COLUMNS)
        ph = ", ".join([self._ph] * len(_JOB_COLUMNS))
        if self.backend == "sqlite":
            return f"INSERT OR REPLACE INTO jobs ({cols}) VALUES ({ph})"
        set_cols = [c for c in _JOB_COLUMNS if c not in ("job_id", "created_at")]
        set_clause = ", ".join(f"{c} = EXCLUDED.{c}" for c in set_cols)
        return f"INSERT INTO jobs ({cols}) VALUES ({ph}) ON CONFLICT (job_id) DO UPDATE SET {set_clause}"

    def _row_to_job(self, row) -> dict:
        return {
            "job_id": row["job_id"],
            "status": row["status"],
            "prompt": row["prompt"],
            "quality_tier": row["quality_tier"],
            "validation": json.loads(row["validation"]),
            "zip_path": row["zip_path"],
            "output_paths": json.loads(row["output_paths"]),
            "error": row["error"],
            "batch_id": row["batch_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        }

    def add_job(
        self,
        job_id: str,
        status: str = "pending",
        prompt: str = "",
        quality_tier: str = "",
        validation=None,
        zip_path: Optional[str] = None,
        output_paths=None,
        error: str = "",
        batch_id: str = "",
    ) -> str:
        now = datetime.utcnow().isoformat() + "Z"
        params = (
            job_id, status, prompt, quality_tier, json.dumps(validation or {}),
            zip_path, json.dumps(output_paths or []), error, batch_id, now, now,
        )
        with self._connection() as conn:
            self._execute(conn, self._insert_job_sql(), params)
        return job_id

    def get_job(self, job_id: str) -> Optional[dict]:
        with self._connection() as conn:
            cur = self._execute(
                conn, f"SELECT * FROM jobs WHERE job_id = {self._ph}", (job_id,)
            )
            row = cur.fetchone()
            return self._row_to_job(row) if row else None

    def update_job(self, job_id: str, **updates) -> Optional[dict]:
        with self._connection() as conn:
            cur = self._execute(
                conn, f"SELECT * FROM jobs WHERE job_id = {self._ph}", (job_id,)
            )
            if cur.fetchone() is None:
                return None
            clean = {}
            for key, value in updates.items():
                if key not in _JOB_UPDATE_COLUMNS:
                    continue
                if key in ("validation", "output_paths"):
                    value = json.dumps(value)
                clean[key] = value
            clean["updated_at"] = datetime.utcnow().isoformat() + "Z"
            set_clause = ", ".join(f"{c} = {self._ph}" for c in clean)
            self._execute(
                conn,
                f"UPDATE jobs SET {set_clause} WHERE job_id = {self._ph}",
                list(clean.values()) + [job_id],
            )
            cur = self._execute(
                conn, f"SELECT * FROM jobs WHERE job_id = {self._ph}", (job_id,)
            )
            return self._row_to_job(cur.fetchone())

    def list_jobs(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        offset: int = 0,
    ) -> List[dict]:
        query = "SELECT * FROM jobs WHERE 1=1"
        params = []
        if status:
            query += f" AND status = {self._ph}"
            params.append(status)
        query += f" ORDER BY created_at DESC LIMIT {self._ph} OFFSET {self._ph}"
        params.extend([limit, offset])
        with self._connection() as conn:
            cur = self._execute(conn, query, params)
            return [self._row_to_job(row) for row in cur.fetchall()]

    def count_jobs(self, status: Optional[str] = None) -> int:
        query = "SELECT COUNT(*) AS count FROM jobs"
        params = []
        if status:
            query += f" WHERE status = {self._ph}"
            params.append(status)
        with self._connection() as conn:
            cur = self._execute(conn, query, params)
            return cur.fetchone()["count"]

    def clear(self):
        with self._connection() as conn:
            self._execute(conn, "DELETE FROM assets")
            self._execute(conn, "DELETE FROM tags")
            self._execute(conn, "DELETE FROM jobs")
