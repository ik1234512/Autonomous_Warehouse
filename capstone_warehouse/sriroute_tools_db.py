"""
sriroute_tools_db.py
====================
SQLite-backed drop-in replacement for sriroute_tools.py's tools.

Same tool names, same argument shapes, same Pydantic schemas (imported
straight from sriroute_tools.py -- the schema doesn't change, only where
the data lives). The difference: instead of reading/writing the
DELIVERY_REQUESTS / FLEET dicts held in memory, these tools read and
write sriroute_fake.db, the SQLite database built by fake_db.py.

Use this exactly like sriroute_tools.py's make_tool_registry -- see
run_db_demo.py for a full ReAct loop wired up against it.

One behavioral difference worth noticing: sriroute_tools.assign_courier
"succeeds" and forgets -- nothing is recorded anywhere. This version's
assign_courier writes a row into the `assignments` table, so a real
audit trail survives after the script exits. Still no capacity/zone/
refrigeration validation, on purpose -- see sriroute_tools.py's
docstring for why that gap is deliberate.
"""

import os
import sqlite3
from typing import Optional

from fake_db import DB_PATH, build_database
from sriroute_tools import TOOL_ARG_MODELS  # noqa: F401 -- re-exported, same schemas apply


def _connect() -> sqlite3.Connection:
    if not os.path.exists(DB_PATH):
        build_database()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def get_delivery_request(request_id: str) -> dict:
    conn = _connect()
    try:
        row = conn.execute(
            "SELECT * FROM delivery_requests WHERE request_id = ?", (request_id,)
        ).fetchone()
    finally:
        conn.close()
    if row is None:
        return {"error": f"{request_id} not found"}
    return dict(row)


def check_fleet_availability(zone: str, vehicle_type: Optional[str] = None) -> dict:
    conn = _connect()
    try:
        if vehicle_type is not None:
            rows = conn.execute(
                "SELECT * FROM fleet WHERE zone = ? AND vehicle_type = ?",
                (zone, vehicle_type),
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT * FROM fleet WHERE zone = ?", (zone,)
            ).fetchall()
    finally:
        conn.close()

    vehicles = [
        {
            "vehicle_id": r["vehicle_id"],
            "vehicle_type": r["vehicle_type"],
            "capacity_kg": r["capacity_kg"],
            "eta_minutes": r["eta_minutes"],
            "priority": bool(r["is_priority"]),
            "surge_cost": r["surge_cost"],
        }
        for r in rows
    ]
    if not vehicles:
        return {"zone": zone, "vehicles": [], "note": "No matching vehicles available in this zone."}
    return {"zone": zone, "vehicles": vehicles}


def assign_courier(request_id: str, vehicle_id: str) -> dict:
    # Deliberately does not validate capacity, zone, or refrigeration --
    # matches sriroute_tools.assign_courier's intentional gap. Unlike
    # that version, this one is persisted: it writes a row to the
    # `assignments` table instead of just returning a message.
    conn = _connect()
    try:
        conn.execute(
            "INSERT INTO assignments (request_id, vehicle_id) VALUES (?, ?)",
            (request_id, vehicle_id),
        )
        conn.commit()
    finally:
        conn.close()
    return {"status": "SUCCESS", "message": f"Assigned {vehicle_id} to {request_id}."}


def default_console_approval(request_id: str, extra_cost: float, reason: str) -> bool:
    answer = input(
        f"[GUARDRAIL] Agent wants to spend an extra Rs.{extra_cost:.2f} on {request_id} "
        f"({reason}). Approve? (y/n): "
    )
    return answer.strip().lower().startswith("y")


def make_request_surge_approval(approval_fn=default_console_approval):
    """Same behavior as sriroute_tools.make_request_surge_approval -- this
    tool doesn't touch the database, so there's nothing to swap out here."""
    def request_surge_approval(request_id: str, extra_cost: float, reason: str) -> dict:
        approved = approval_fn(request_id, extra_cost, reason)
        if approved:
            return {"status": "APPROVED", "request_id": request_id, "extra_cost": extra_cost}
        return {
            "status": "DENIED", "request_id": request_id,
            "note": "Supervisor denied the surge request. Use the standard fleet "
                    "and notify the customer of a possible delay.",
        }
    return request_surge_approval


def make_tool_registry(approval_fn=None):
    """SQLite-backed equivalent of sriroute_tools.make_tool_registry."""
    surge_fn = make_request_surge_approval(approval_fn) if approval_fn else make_request_surge_approval()
    return {
        "get_delivery_request": get_delivery_request,
        "check_fleet_availability": check_fleet_availability,
        "assign_courier": assign_courier,
        "request_surge_approval": surge_fn,
    }
