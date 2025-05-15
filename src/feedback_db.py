# SQLite feedback DB and logging

# src/feedback_db.py

import sqlite3
from sqlite3 import Connection, Cursor
from typing import Optional, List, Tuple, Dict, Any
import datetime


class FeedbackDB:
    """
    A simple SQLite-based feedback database for storing user votes and refinement feedback on LLM explanations.
    """

    def __init__(self, db_path: str = "logs/feedback.db"):
        self.db_path = db_path
        self.conn: Optional[Connection] = None
        self._connect()
        self._create_tables()

    def _connect(self):
        """Establish a connection to the SQLite database."""
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row

    def _create_tables(self):
        """Create feedback tables if they don't exist."""
        with self.conn:
            self.conn.execute(
                """
                CREATE TABLE IF NOT EXISTS feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    explanation_id TEXT,
                    vote INTEGER, -- 1=helpful, 0=not helpful
                    refinement TEXT,
                    timestamp TEXT
                )
                """
            )

    def insert_feedback(
        self,
        user_id: str,
        explanation_id: str,
        vote: int,
        refinement: Optional[str] = None,
    ):
        """
        Insert a feedback record.

        Args:
            user_id: Identifier for the user providing feedback
            explanation_id: Unique ID of the explanation the feedback relates to
            vote: Integer vote (e.g., 1=helpful, 0=not helpful)
            refinement: Optional text refinement provided by the user
        """
        timestamp = datetime.datetime.utcnow().isoformat()
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO feedback (user_id, explanation_id, vote, refinement, timestamp)
                VALUES (?, ?, ?, ?, ?)
                """,
                (user_id, explanation_id, vote, refinement, timestamp),
            )

    def get_feedback(
        self,
        explanation_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve feedback records, optionally filtered by explanation or user.

        Args:
            explanation_id: Filter feedback by explanation ID (optional)
            user_id: Filter feedback by user ID (optional)

        Returns:
            List of feedback dictionaries
        """
        query = "SELECT * FROM feedback WHERE 1=1"
        params = []
        if explanation_id:
            query += " AND explanation_id = ?"
            params.append(explanation_id)
        if user_id:
            query += " AND user_id = ?"
            params.append(user_id)

        cursor = self.conn.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def get_vote_stats(self, explanation_id: str) -> Dict[str, Any]:
        """
        Get aggregated vote statistics for a given explanation.

        Args:
            explanation_id: Explanation ID to aggregate stats on.

        Returns:
            Dictionary with counts of helpful and not helpful votes.
        """
        cursor = self.conn.execute(
            """
            SELECT
                SUM(CASE WHEN vote = 1 THEN 1 ELSE 0 END) AS helpful,
                SUM(CASE WHEN vote = 0 THEN 1 ELSE 0 END) AS not_helpful,
                COUNT(*) AS total
            FROM feedback
            WHERE explanation_id = ?
            """,
            (explanation_id,),
        )
        row = cursor.fetchone()
        if row:
            return {
                "helpful": row["helpful"] or 0,
                "not_helpful": row["not_helpful"] or 0,
                "total": row["total"] or 0,
            }
        else:
            return {"helpful": 0, "not_helpful": 0, "total": 0}

    def close(self):
        """Close the database connection."""
        if self.conn:
            self.conn.close()
            self.conn = None


if __name__ == "__main__":
    # Simple test
    db = FeedbackDB()
    db.insert_feedback(user_id="user123", explanation_id="exp456", vote=1, refinement="More detail on X")
    feedbacks = db.get_feedback()
    print(feedbacks)
    stats = db.get_vote_stats("exp456")
    print(stats)
    db.close()
