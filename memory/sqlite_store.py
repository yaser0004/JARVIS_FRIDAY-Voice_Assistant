from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from core.config import DB_FILE


class SQLiteStore:
    def __init__(self, db_path: Path = DB_FILE) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_tables()

    def _init_tables(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS conversation_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                role TEXT NOT NULL,
                text TEXT NOT NULL,
                intent TEXT,
                confidence REAL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS user_preferences (
                key TEXT PRIMARY KEY,
                value TEXT,
                updated_at REAL NOT NULL
            )
            """
        )
        cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS app_usage_stats (
                app_name TEXT PRIMARY KEY,
                launch_count INTEGER NOT NULL DEFAULT 0,
                last_used REAL NOT NULL
            )
            """
        )
        self._conn.commit()

    def save_turn(
        self, role: str, text: str, intent: Optional[str] = None, confidence: Optional[float] = None
    ) -> None:
        self._conn.execute(
            """
            INSERT INTO conversation_history(timestamp, role, text, intent, confidence)
            VALUES (?, ?, ?, ?, ?)
            """,
            (time.time(), role, text, intent, confidence),
        )
        self._conn.commit()

    def get_history(self, limit: int = 50) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT id, timestamp, role, text, intent, confidence
            FROM conversation_history
            ORDER BY id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def set_preference(self, key: str, value: str) -> None:
        self._conn.execute(
            """
            INSERT INTO user_preferences(key, value, updated_at)
            VALUES (?, ?, ?)
            ON CONFLICT(key)
            DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at
            """,
            (key, value, time.time()),
        )
        self._conn.commit()

    def get_preference(self, key: str, default: Any = None) -> Any:
        row = self._conn.execute(
            "SELECT value FROM user_preferences WHERE key = ?",
            (key,),
        ).fetchone()
        if row is None:
            return default
        return row["value"]

    def increment_app_usage(self, app_name: str) -> None:
        self._conn.execute(
            """
            INSERT INTO app_usage_stats(app_name, launch_count, last_used)
            VALUES (?, 1, ?)
            ON CONFLICT(app_name)
            DO UPDATE SET launch_count = launch_count + 1, last_used = excluded.last_used
            """,
            (app_name, time.time()),
        )
        self._conn.commit()

    def get_frequent_apps(self, n: int = 5) -> List[Dict[str, Any]]:
        rows = self._conn.execute(
            """
            SELECT app_name, launch_count, last_used
            FROM app_usage_stats
            ORDER BY launch_count DESC, last_used DESC
            LIMIT ?
            """,
            (n,),
        ).fetchall()
        return [dict(row) for row in rows]
