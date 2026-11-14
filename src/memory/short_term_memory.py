"""
src/memory/short_term_memory.py
Task 3: Short-Term Relational Memory Store (SQLite)
Implements:
1. Session-keyed storage for ordered conversational turns
2. At least 4 structured slots:
   - policy_number
   - product_type
   - claim_reference
   - city
   (plus auxiliary slots like car_engine_cc, incident_date)
3. Exact slot lookup and slot overwrite on user correction
"""

import sqlite3
import os
import json
from typing import Dict, Any, List, Optional
from datetime import datetime


class ShortTermMemoryStore:
    def __init__(self, db_path: str = "data/memory_short_term.db"):
        """Initializes SQLite database and tables for sessions, turns, and slots."""
        self.db_path = db_path
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns SQLite connection with row factory enabled."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        """Creates the relational schema for turns and structured slots."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            # Table 1: Turn history (ordered sequence per session)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    turn_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    session_id TEXT NOT NULL,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)

            # Table 2: Structured slots (exact key-value state per session)
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS session_slots (
                    session_id TEXT NOT NULL,
                    slot_name TEXT NOT NULL,
                    slot_value TEXT NOT NULL,
                    updated_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (session_id, slot_name)
                );
            """)

            # Table 3: Session metadata and consolidated summary
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sessions (
                    session_id TEXT PRIMARY KEY,
                    user_id TEXT NOT NULL,
                    summary TEXT DEFAULT '',
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                );
            """)
            conn.commit()

    # -------------------------------------------------------------
    # 1. TURN RECORDING & RETRIEVAL
    # -------------------------------------------------------------
    def record_turn(self, session_id: str, role: str, content: str, user_id: str = "default_user"):
        """Records an ordered turn into the session history."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO sessions (session_id, user_id)
                VALUES (?, ?);
            """, (session_id, user_id))

            cursor.execute("""
                INSERT INTO conversation_turns (session_id, role, content)
                VALUES (?, ?, ?);
            """, (session_id, role, content))
            conn.commit()

    def get_recent_turns(self, session_id: str, limit: int = 6) -> List[Dict[str, str]]:
        """Retrieves the most recent ordered turns for the active session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT role, content, timestamp FROM conversation_turns
                WHERE session_id = ?
                ORDER BY turn_id DESC
                LIMIT ?;
            """, (session_id, limit))
            rows = cursor.fetchall()
            turns = [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]
            return turns

    def get_turn_count(self, session_id: str) -> int:
        """Returns total turns elapsed in this session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(*) as count FROM conversation_turns WHERE session_id = ?", (session_id,))
            return cursor.fetchone()["count"]

    # -------------------------------------------------------------
    # 2. STRUCTURED SLOTS (EXACT LOOKUP & OVERWRITE)
    # -------------------------------------------------------------
    def set_slot(self, session_id: str, slot_name: str, slot_value: str):
        """Sets or updates a structured slot. Newer values overwrite previous ones."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO session_slots (session_id, slot_name, slot_value, updated_at)
                VALUES (?, ?, ?, CURRENT_TIMESTAMP)
                ON CONFLICT(session_id, slot_name) DO UPDATE SET
                    slot_value = excluded.slot_value,
                    updated_at = CURRENT_TIMESTAMP;
            """, (session_id, slot_name, slot_value))
            conn.commit()

    def get_slot(self, session_id: str, slot_name: str) -> Optional[str]:
        """Performs exact lookup of a specific slot by name."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT slot_value FROM session_slots
                WHERE session_id = ? AND slot_name = ?;
            """, (session_id, slot_name))
            row = cursor.fetchone()
            return row["slot_value"] if row else None

    def get_all_slots(self, session_id: str) -> Dict[str, str]:
        """Returns all populated slots for the session as a clean dictionary."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                SELECT slot_name, slot_value FROM session_slots
                WHERE session_id = ?;
            """, (session_id,))
            rows = cursor.fetchall()
            return {r["slot_name"]: r["slot_value"] for r in rows}

    # -------------------------------------------------------------
    # 3. CONSOLIDATED SUMMARY STORAGE
    # -------------------------------------------------------------
    def update_session_summary(self, session_id: str, summary: str):
        """Stores consolidated summary of older turns."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("""
                UPDATE sessions SET summary = ? WHERE session_id = ?;
            """, (summary, session_id))
            conn.commit()

    def get_session_summary(self, session_id: str) -> str:
        """Retrieves consolidated summary for the session."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT summary FROM sessions WHERE session_id = ?;", (session_id,))
            row = cursor.fetchone()
            return row["summary"] if row and row["summary"] else ""