"""SQLite database layer for analysis history, users, and training records."""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Generator, Optional

_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "biomarker.db"
_local = threading.local()


# ---------------------------------------------------------------------------
# Connection helpers
# ---------------------------------------------------------------------------

def _get_connection() -> sqlite3.Connection:
    """Return a thread-local SQLite connection, creating it if needed."""
    conn: Optional[sqlite3.Connection] = getattr(_local, "conn", None)
    if conn is None:
        _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(_DB_PATH), check_same_thread=False)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        _local.conn = conn
    return conn


@contextmanager
def get_db() -> Generator[sqlite3.Connection, None, None]:
    """Context manager that yields a connection and commits on success."""
    conn = _get_connection()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise


def init_db() -> None:
    """Create all tables if they do not exist."""
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS users (
                id          TEXT PRIMARY KEY,
                username    TEXT UNIQUE NOT NULL,
                password_hash TEXT NOT NULL,
                role        TEXT NOT NULL DEFAULT 'clinician',
                is_active   INTEGER NOT NULL DEFAULT 1,
                created_at  TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS analyses (
                id              TEXT PRIMARY KEY,
                user_id         TEXT,
                file_id         TEXT,
                file_path       TEXT,
                filename        TEXT,
                status          TEXT NOT NULL DEFAULT 'pending',
                features        TEXT DEFAULT '{}',
                biomarkers      TEXT DEFAULT '[]',
                risk_score      REAL,
                recommendations TEXT DEFAULT '[]',
                processing_time_ms REAL,
                error           TEXT,
                created_at      TEXT NOT NULL,
                completed_at    TEXT
            );

            CREATE TABLE IF NOT EXISTS training_history (
                id          TEXT PRIMARY KEY,
                model_type  TEXT NOT NULL,
                epochs      INTEGER,
                learning_rate REAL,
                batch_size  INTEGER,
                status      TEXT NOT NULL DEFAULT 'pending',
                progress    REAL DEFAULT 0.0,
                current_epoch INTEGER DEFAULT 0,
                loss        REAL,
                accuracy    REAL,
                started_at  TEXT,
                completed_at TEXT,
                error       TEXT,
                extra_params TEXT DEFAULT '{}'
            );

            CREATE INDEX IF NOT EXISTS idx_analyses_user ON analyses(user_id);
            CREATE INDEX IF NOT EXISTS idx_analyses_status ON analyses(status);
            CREATE INDEX IF NOT EXISTS idx_training_status ON training_history(status);
            """
        )


# ---------------------------------------------------------------------------
# User CRUD
# ---------------------------------------------------------------------------

def create_user(
    username: str,
    password_hash: str,
    role: str = "clinician",
) -> dict[str, Any]:
    user_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO users (id, username, password_hash, role, is_active, created_at) "
            "VALUES (?, ?, ?, ?, 1, ?)",
            (user_id, username, password_hash, role, now),
        )
    return get_user(user_id)  # type: ignore[return-value]


def get_user(user_id: str) -> Optional[dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute("SELECT * FROM users WHERE id = ?", (user_id,)).fetchone()
    return dict(row) if row else None


def get_user_by_username(username: str) -> Optional[dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM users WHERE username = ?", (username,)
        ).fetchone()
    return dict(row) if row else None


def list_users() -> list[dict[str, Any]]:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT id, username, role, is_active, created_at FROM users ORDER BY created_at DESC"
        ).fetchall()
    return [dict(r) for r in rows]


# ---------------------------------------------------------------------------
# Analysis CRUD
# ---------------------------------------------------------------------------

def create_analysis(
    *,
    user_id: Optional[str] = None,
    file_id: Optional[str] = None,
    file_path: Optional[str] = None,
    filename: Optional[str] = None,
    status: str = "pending",
) -> dict[str, Any]:
    analysis_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO analyses "
            "(id, user_id, file_id, file_path, filename, status, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (analysis_id, user_id, file_id, file_path, filename, status, now),
        )
    return get_analysis(analysis_id)  # type: ignore[return-value]


def get_analysis(analysis_id: str) -> Optional[dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM analyses WHERE id = ?", (analysis_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["features"] = json.loads(d.get("features") or "{}")
    d["biomarkers"] = json.loads(d.get("biomarkers") or "[]")
    d["recommendations"] = json.loads(d.get("recommendations") or "[]")
    return d


def update_analysis(analysis_id: str, **fields: Any) -> None:
    if not fields:
        return
    set_parts: list[str] = []
    values: list[Any] = []
    for key, val in fields.items():
        if key in ("features", "biomarkers", "recommendations", "extra_params"):
            val = json.dumps(val)
        set_parts.append(f"{key} = ?")
        values.append(val)
    values.append(analysis_id)
    with get_db() as conn:
        conn.execute(
            f"UPDATE analyses SET {', '.join(set_parts)} WHERE id = ?", values
        )


def list_analyses(
    *,
    user_id: Optional[str] = None,
    page: int = 1,
    page_size: int = 20,
) -> tuple[list[dict[str, Any]], int]:
    """Return (items, total_count)."""
    where = "WHERE 1=1"
    params: list[Any] = []
    if user_id:
        where += " AND user_id = ?"
        params.append(user_id)

    with get_db() as conn:
        total = conn.execute(
            f"SELECT COUNT(*) as cnt FROM analyses {where}", params
        ).fetchone()["cnt"]

        offset = (page - 1) * page_size
        rows = conn.execute(
            f"SELECT * FROM analyses {where} ORDER BY created_at DESC LIMIT ? OFFSET ?",
            params + [page_size, offset],
        ).fetchall()

    items: list[dict[str, Any]] = []
    for row in rows:
        d = dict(row)
        d["features"] = json.loads(d.get("features") or "{}")
        d["biomarkers"] = json.loads(d.get("biomarkers") or "[]")
        d["recommendations"] = json.loads(d.get("recommendations") or "[]")
        items.append(d)
    return items, total


def delete_analysis(analysis_id: str) -> bool:
    with get_db() as conn:
        cursor = conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    return cursor.rowcount > 0


# ---------------------------------------------------------------------------
# Training CRUD
# ---------------------------------------------------------------------------

def create_training_record(
    *,
    model_type: str = "biomarker_classifier",
    epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 32,
    extra_params: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    record_id = uuid.uuid4().hex
    now = datetime.utcnow().isoformat()
    with get_db() as conn:
        conn.execute(
            "INSERT INTO training_history "
            "(id, model_type, epochs, learning_rate, batch_size, status, started_at, extra_params) "
            "VALUES (?, ?, ?, ?, ?, 'training', ?, ?)",
            (record_id, model_type, epochs, learning_rate, batch_size, now,
             json.dumps(extra_params or {})),
        )
    return get_training_record(record_id)  # type: ignore[return-value]


def get_training_record(record_id: str) -> Optional[dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM training_history WHERE id = ?", (record_id,)
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["extra_params"] = json.loads(d.get("extra_params") or "{}")
    return d


def get_active_training() -> Optional[dict[str, Any]]:
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM training_history WHERE status = 'training' "
            "ORDER BY started_at DESC LIMIT 1"
        ).fetchone()
    if row is None:
        return None
    d = dict(row)
    d["extra_params"] = json.loads(d.get("extra_params") or "{}")
    return d


def update_training_record(record_id: str, **fields: Any) -> None:
    if not fields:
        return
    set_parts: list[str] = []
    values: list[Any] = []
    for key, val in fields.items():
        if key == "extra_params":
            val = json.dumps(val)
        set_parts.append(f"{key} = ?")
        values.append(val)
    values.append(record_id)
    with get_db() as conn:
        conn.execute(
            f"UPDATE training_history SET {', '.join(set_parts)} WHERE id = ?",
            values,
        )
