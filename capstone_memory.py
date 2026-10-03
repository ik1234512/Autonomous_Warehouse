"""SQLite-backed history for the capstone warehouse run."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


class WarehouseMemory:
    def __init__(self, db_path: str | Path = "warehouse_memory.db") -> None:
        self.db_path = str(db_path)
        # The dashboard and worker thread both use this connection, so keep it
        # thread-safe and serialize the DB writes.
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._create_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def __enter__(self) -> "WarehouseMemory":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _create_schema(self) -> None:
        self._conn.executescript(
            """
            PRAGMA foreign_keys = ON;

            CREATE TABLE IF NOT EXISTS runs (
                run_id INTEGER PRIMARY KEY AUTOINCREMENT,
                started_at TEXT NOT NULL,
                label TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                recorded_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS robot_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                recorded_at TEXT NOT NULL,
                robot_id TEXT NOT NULL,
                x INTEGER,
                y INTEGER,
                estimated_x INTEGER,
                estimated_y INTEGER,
                battery REAL,
                wear REAL,
                status TEXT,
                current_task TEXT,
                carried_item TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS task_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                recorded_at TEXT NOT NULL,
                task_id TEXT NOT NULL,
                item_id TEXT,
                status TEXT,
                priority TEXT,
                deadline TEXT,
                assigned_robot_id TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE TABLE IF NOT EXISTS item_snapshots (
                snapshot_id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER,
                recorded_at TEXT NOT NULL,
                item_id TEXT NOT NULL,
                weight REAL,
                x INTEGER,
                y INTEGER,
                status TEXT,
                FOREIGN KEY (run_id) REFERENCES runs(run_id)
            );

            CREATE INDEX IF NOT EXISTS idx_events_run_time
                ON events(run_id, recorded_at);
            CREATE INDEX IF NOT EXISTS idx_events_type
                ON events(event_type);
            CREATE INDEX IF NOT EXISTS idx_robot_history
                ON robot_snapshots(run_id, robot_id, recorded_at);
            CREATE INDEX IF NOT EXISTS idx_task_history
                ON task_snapshots(run_id, task_id, recorded_at);
            """
        )
        self._conn.commit()

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    def start_run(self, label: str = "warehouse-capstone") -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO runs(started_at, label) VALUES (?, ?)",
                (self._now(), label),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def record_event(
        self,
        event_type: str,
        payload: dict[str, Any] | None = None,
        run_id: int | None = None,
    ) -> int:
        payload = payload or {}
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO events(run_id, recorded_at, event_type, payload_json)
                VALUES (?, ?, ?, ?)
                """,
                (
                    run_id,
                    self._now(),
                    event_type,
                    json.dumps(payload, default=self._json_default, sort_keys=True),
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid)

    def snapshot_state(self, state, run_id: int | None = None) -> None:
        """Persist current warehouse state without changing the state object."""
        recorded_at = self._now()
        with self._lock:
            conn = self._conn
            with conn:
                for robot in state.robots.values():
                    pos = getattr(robot, "position", None)
                    est = getattr(robot, "estimated_position", None)
                    conn.execute(
                        """
                        INSERT INTO robot_snapshots(
                            run_id, recorded_at, robot_id, x, y,
                            estimated_x, estimated_y, battery, wear,
                            status, current_task, carried_item
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            recorded_at,
                            str(getattr(robot, "robot_id", "")),
                            self._coord(pos, 0),
                            self._coord(pos, 1),
                            self._coord(est, 0),
                            self._coord(est, 1),
                            self._number(getattr(robot, "battery", None)),
                            self._number(getattr(robot, "wear", None)),
                            self._text(getattr(robot, "status", None)),
                            self._text(getattr(robot, "current_task", None)),
                            self._text(getattr(robot, "carried_item", None)),
                        ),
                    )

                for task in state.tasks.values():
                    deadline = getattr(task, "deadline", None)
                    conn.execute(
                        """
                        INSERT INTO task_snapshots(
                            run_id, recorded_at, task_id, item_id, status,
                            priority, deadline, assigned_robot_id
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            recorded_at,
                            str(getattr(task, "task_id", "")),
                            self._text(getattr(task, "item_id", None)),
                            self._text(getattr(task, "status", None)),
                            self._text(getattr(task, "priority", None)),
                            self._text(deadline),
                            self._text(getattr(task, "assigned_robot_id", None)),
                        ),
                    )

                for item in state.inventory.values():
                    pos = getattr(item, "location", None)
                    conn.execute(
                        """
                        INSERT INTO item_snapshots(
                            run_id, recorded_at, item_id, weight, x, y, status
                        ) VALUES (?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            run_id,
                            recorded_at,
                            str(getattr(item, "item_id", "")),
                            self._number(getattr(item, "weight", None)),
                            self._coord(pos, 0),
                            self._coord(pos, 1),
                            self._text(getattr(item, "status", None)),
                        ),
                    )

    def attach_to_state_log(self, state, run_id: int) -> Callable[..., Any]:
        """
        Wrap state.log_event at runtime.

        The original logger still runs. The wrapper additionally persists the
        same event and a full state snapshot, making the DB an audit trail.
        """
        original = state.log_event

        def wrapped(event_type: str, **kwargs: Any) -> Any:
            result = original(event_type, **kwargs)
            self.record_event(event_type, kwargs, run_id=run_id)
            self.snapshot_state(state, run_id=run_id)
            return result

        state.log_event = wrapped
        return original

    def recent_events(self, run_id: int, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT event_id, recorded_at, event_type, payload_json
                FROM events
                WHERE run_id = ?
                ORDER BY event_id DESC
                LIMIT ?
                """,
                (run_id, limit),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"],
                "recorded_at": row["recorded_at"],
                "event_type": row["event_type"],
                "payload": json.loads(row["payload_json"]),
            }
            for row in rows
        ]

    def robot_history(self, run_id: int, robot_id: int | str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT recorded_at, robot_id, x, y, estimated_x, estimated_y,
                       battery, wear, status, current_task, carried_item
                FROM robot_snapshots
                WHERE run_id = ? AND robot_id = ?
                ORDER BY snapshot_id
                """,
                (run_id, str(robot_id)),
            ).fetchall()
        return [dict(row) for row in rows]

    def task_history(self, run_id: int, task_id: str) -> list[dict[str, Any]]:
        with self._lock:
            rows = self._conn.execute(
                """
                SELECT recorded_at, task_id, item_id, status, priority,
                       deadline, assigned_robot_id
                FROM task_snapshots
                WHERE run_id = ? AND task_id = ?
                ORDER BY snapshot_id
                """,
                (run_id, task_id),
            ).fetchall()
        return [dict(row) for row in rows]

    def count_events(self, run_id: int) -> int:
        with self._lock:
            row = self._conn.execute(
                "SELECT COUNT(*) AS n FROM events WHERE run_id = ?",
                (run_id,),
            ).fetchone()
        return int(row["n"])

    @staticmethod
    def _coord(value: Any, index: int) -> int | None:
        try:
            return int(value[index])
        except (TypeError, IndexError, KeyError, ValueError):
            return None

    @staticmethod
    def _number(value: Any) -> float | None:
        try:
            return float(value) if value is not None else None
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _text(value: Any) -> str | None:
        return None if value is None else str(value)

    @staticmethod
    def _json_default(value: Any) -> Any:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "__dict__"):
            return value.__dict__
        if isinstance(value, set):
            return sorted(value)
        return str(value)
