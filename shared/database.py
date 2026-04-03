"""
Database layer - SQLite for dev, designed to swap to Postgres.

I started with raw sqlite3 but switched to a thin wrapper because
managing connections and cursors manually was error-prone during
the scoring→anti-cheat pipeline where multiple modules read/write
concurrently.
"""

import sqlite3
import json
import os
from datetime import datetime
from typing import List, Optional, Dict, Any
from contextlib import contextmanager

DB_PATH = os.environ.get("GTHR_DB_PATH", "gthr_hiring.db")


class Database:
    def __init__(self, db_path: str = None):
        self.db_path = db_path or DB_PATH
        self._init_tables()

    def _init_tables(self):
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS candidates (
                    candidate_id TEXT PRIMARY KEY,
                    name TEXT,
                    email TEXT,
                    role_applied_for TEXT,
                    profile_json TEXT,
                    metadata_json TEXT,
                    ingested_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(email, role_applied_for)
                );

                CREATE TABLE IF NOT EXISTS answers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT,
                    question_id TEXT,
                    question_text TEXT,
                    answer_text TEXT,
                    submitted_at TEXT,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
                );

                CREATE TABLE IF NOT EXISTS scores (
                    candidate_id TEXT PRIMARY KEY,
                    total_score REAL,
                    dimension_scores_json TEXT,
                    tier TEXT,
                    explanation_json TEXT,
                    scored_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
                );

                CREATE TABLE IF NOT EXISTS cheat_flags (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT,
                    flag_type TEXT,
                    details TEXT,
                    severity REAL,
                    flagged_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
                );

                CREATE TABLE IF NOT EXISTS interaction_logs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    candidate_id TEXT,
                    event_type TEXT,
                    event_data TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
                );

                CREATE TABLE IF NOT EXISTS email_threads (
                    thread_id TEXT PRIMARY KEY,
                    candidate_id TEXT,
                    status TEXT DEFAULT 'initiated',
                    round INTEGER DEFAULT 1,
                    last_email_at TEXT,
                    history_json TEXT,
                    FOREIGN KEY (candidate_id) REFERENCES candidates(candidate_id)
                );

                CREATE TABLE IF NOT EXISTS learning_insights (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    insight_type TEXT,
                    insight_data TEXT,
                    batch_size INTEGER,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS weight_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    weights_json TEXT,
                    reason TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );
            """)

    @contextmanager
    def _conn(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def upsert_candidate(self, candidate_data: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO candidates (candidate_id, name, email, role_applied_for, profile_json, metadata_json)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(email, role_applied_for) DO UPDATE SET
                    name = excluded.name,
                    profile_json = excluded.profile_json,
                    metadata_json = excluded.metadata_json
            """, (
                candidate_data["candidate_id"],
                candidate_data["name"],
                candidate_data["email"],
                candidate_data["role_applied_for"],
                json.dumps(candidate_data.get("profile", {})),
                json.dumps(candidate_data.get("metadata", {}))
            ))

            # Insert answers
            for answer in candidate_data.get("answers", []):
                conn.execute("""
                    INSERT OR REPLACE INTO answers (candidate_id, question_id, question_text, answer_text, submitted_at)
                    VALUES (?, ?, ?, ?, ?)
                """, (
                    candidate_data["candidate_id"],
                    answer["question_id"],
                    answer["question_text"],
                    answer["answer_text"],
                    answer.get("submitted_at", datetime.utcnow().isoformat())
                ))

    def save_score(self, candidate_id: str, score_data: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT OR REPLACE INTO scores (candidate_id, total_score, dimension_scores_json, tier, explanation_json)
                VALUES (?, ?, ?, ?, ?)
            """, (
                candidate_id,
                score_data["total_score"],
                json.dumps(score_data["dimension_scores"]),
                score_data["tier"],
                json.dumps(score_data["explanation"])
            ))

    def save_cheat_flag(self, candidate_id: str, flag_type: str, details: str, severity: float):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO cheat_flags (candidate_id, flag_type, details, severity)
                VALUES (?, ?, ?, ?)
            """, (candidate_id, flag_type, details, severity))

    def log_interaction(self, candidate_id: str, event_type: str, event_data: dict):
        with self._conn() as conn:
            conn.execute("""
                INSERT INTO interaction_logs (candidate_id, event_type, event_data)
                VALUES (?, ?, ?)
            """, (candidate_id, event_type, json.dumps(event_data)))

    def get_all_candidates(self) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM candidates").fetchall()
            return [dict(r) for r in rows]

    def get_all_flags(self, candidate_id: str) -> List[dict]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM cheat_flags WHERE candidate_id = ?", (candidate_id,)
            ).fetchall()
            return [dict(r) for r in rows]

    def get_flag_count(self, candidate_id: str) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) as cnt FROM cheat_flags WHERE candidate_id = ?", (candidate_id,)
            ).fetchone()
            return row["cnt"]
