from __future__ import annotations

import json
import sqlite3
import threading
import time
from collections import OrderedDict
from contextlib import closing
from pathlib import Path
from typing import Any

from config import MAX_HISTORY_MESSAGES, MAX_SESSIONS, SESSION_TTL_SECONDS


class SessionStore:
    """Small thread-safe in-memory session store with TTL cleanup."""

    def __init__(
        self,
        *,
        ttl_seconds: int = SESSION_TTL_SECONDS,
        max_sessions: int = MAX_SESSIONS,
        max_history_messages: int = MAX_HISTORY_MESSAGES,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_history_messages = max_history_messages
        self._lock = threading.RLock()
        self._items: OrderedDict[str, dict[str, Any]] = OrderedDict()

    @staticmethod
    def _empty() -> dict[str, Any]:
        return {
            "history": [],
            "last_question": None,
            "last_response": None,
            "last_passed": False,
            "updated_at": time.time(),
        }

    def _cleanup_locked(self) -> None:
        now = time.time()
        expired = [sid for sid, item in self._items.items() if now - item.get("updated_at", now) > self.ttl_seconds]
        for sid in expired:
            self._items.pop(sid, None)
        while len(self._items) > self.max_sessions:
            self._items.popitem(last=False)

    def get(self, session_id: str) -> dict[str, Any]:
        with self._lock:
            self._cleanup_locked()
            item = self._items.get(session_id)
            if not item:
                return self._empty()
            self._items.move_to_end(session_id)
            return {
                "history": list(item.get("history", [])),
                "last_question": item.get("last_question"),
                "last_response": item.get("last_response"),
                "last_passed": bool(item.get("last_passed", False)),
                "updated_at": item.get("updated_at", time.time()),
            }

    def update(self, session_id: str, question: str, answer: str, passed: bool) -> None:
        with self._lock:
            item = self._items.get(session_id, self._empty())
            history = list(item.get("history", []))
            history.append({"role": "user", "content": question})
            history.append({"role": "assistant", "content": answer})
            self._items[session_id] = {
                "history": history[-self.max_history_messages :],
                "last_question": question,
                "last_response": answer,
                "last_passed": passed,
                "updated_at": time.time(),
            }
            self._items.move_to_end(session_id)
            self._cleanup_locked()

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._items[session_id] = self._empty()
            self._items.move_to_end(session_id)

    def end(self, session_id: str) -> None:
        with self._lock:
            self._items.pop(session_id, None)

    def count(self) -> int:
        with self._lock:
            self._cleanup_locked()
            return len(self._items)


class FeedbackStore:
    """SQLite-backed feedback writer. Safe for concurrent requests."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.db_path, timeout=30, isolation_level=None)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        return conn

    def _init_db(self) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    feedback TEXT NOT NULL,
                    suggestion TEXT NOT NULL DEFAULT '',
                    created_at TEXT NOT NULL,
                    metadata_json TEXT NOT NULL DEFAULT '{}'
                )
                """
            )
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_session ON feedback(session_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_feedback_created ON feedback(created_at)")
            conn.commit()

    def add(self, data: dict[str, Any]) -> None:
        with self._lock, closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO feedback (
                    session_id, question, answer, feedback, suggestion, created_at, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    data.get("session_id", ""),
                    data.get("question", ""),
                    data.get("answer", ""),
                    data.get("feedback", ""),
                    data.get("suggestion", ""),
                    data.get("timestamp", ""),
                    json.dumps(data.get("metadata", {}), ensure_ascii=False),
                ),
            )
            conn.commit()

    def healthy(self) -> bool:
        try:
            with closing(self._connect()) as conn:
                conn.execute("SELECT 1")
            return True
        except sqlite3.Error:
            return False
