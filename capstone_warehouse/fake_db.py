"""
fake_db.py
====================
A tiny FAKE database for the SriRoute lab, built with nothing but the
Python standard library's `sqlite3` module. It's "fake" in the sense that
it's throwaway demo data (same flavor as sriroute_tools.py's in-memory
dicts) -- but it's a REAL SQLite database file on disk, created and
queried with real SQL.

This does not modify sriroute_tools.py. It's the storage layer behind
sriroute_tools_db.py, a SQLite-backed drop-in replacement for that
module's tools -- see run_db_demo.py for a full ReAct loop run against
this database instead of the in-memory dicts.

Run this file directly to (re)create the database and see it work:

    python3 fake_db.py

It will:
  1. Create sriroute_fake.db (deleting any old copy first)
  2. Create three tables: delivery_requests, fleet, and assignments
  3. Insert some rows
  4. Run a few example SELECT queries
  5. Run one UPDATE and one DELETE to show those too
"""

import sqlite3
import os

DB_PATH = "sriroute_fake.db"


def build_database(db_path: str = DB_PATH) -> None:
    """Create the database file from scratch and populate it."""
    if os.path.exists(db_path):
        os.remove(db_path)  # start clean every run

    conn = sqlite3.connect(db_path)
    cur = conn.cursor()

    # --- 1. CREATE TABLE ---------------------------------------------
    cur.execute("""
        CREATE TABLE delivery_requests (
            request_id      TEXT PRIMARY KEY,
            customer        TEXT NOT NULL,
            pickup          TEXT NOT NULL,
            dropoff         TEXT NOT NULL,
            package_type    TEXT NOT NULL,
            weight_kg       REAL NOT NULL,
            zone            TEXT NOT NULL,
            promised_window TEXT NOT NULL
        )
    """)

    cur.execute("""
        CREATE TABLE fleet (
            vehicle_id   TEXT PRIMARY KEY,
            vehicle_type TEXT NOT NULL,
            zone         TEXT NOT NULL,
            capacity_kg  REAL NOT NULL,
            eta_minutes  INTEGER NOT NULL,
            is_priority  INTEGER NOT NULL DEFAULT 0,   -- SQLite has no
            surge_cost   REAL NOT NULL DEFAULT 0.0     -- real boolean type
        )
    """)

    cur.execute("""
        CREATE TABLE assignments (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id  TEXT NOT NULL,
            vehicle_id  TEXT NOT NULL,
            assigned_at TEXT NOT NULL DEFAULT (datetime('now'))
        )
    """)

    # --- 2. INSERT rows -------------------------------------------------
    delivery_rows = [
        ("REQ-901", "Meera",   "T Nagar",       "Adyar",          "standard",   4,  "South Chennai", "2:00-3:00 PM"),
        ("REQ-902", "Farhan",  "OMR Warehouse", "Sholinganallur", "perishable", 12, "OMR",            "1:00-1:30 PM"),
        ("REQ-903", "Divya",   "Sri City Hub",  "Tada",           "standard",   25, "Sri City",       "4:00-6:00 PM"),
        ("REQ-904", "Karthik", "Sri City Hub",  "Tada",           "standard",   5,  "Sri City",       "11:00-11:15 AM"),
    ]
    cur.executemany(
        "INSERT INTO delivery_requests VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        delivery_rows,
    )

    fleet_rows = [
        ("BIKE-12",         "bike", "South Chennai", 8,  10, 0, 0.0),
        ("VAN-04",          "van",  "South Chennai", 50, 18, 0, 0.0),
        ("BIKE-07",         "bike", "OMR",           8,  6,  0, 0.0),
        ("VAN-09",          "van",  "Sri City",      60, 25, 0, 0.0),
        ("PRIORITY-VAN-01", "van",  "Sri City",      60, 8,  1, 450.0),
    ]
    cur.executemany(
        "INSERT INTO fleet VALUES (?, ?, ?, ?, ?, ?, ?)",
        fleet_rows,
    )

    conn.commit()  # writes the inserts to disk
    conn.close()


def demo_queries(db_path: str = DB_PATH) -> None:
    """Run a handful of example SELECT / UPDATE / DELETE statements."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row  # lets us read columns by name
    cur = conn.cursor()

    print("\n--- All delivery requests ---")
    for row in cur.execute("SELECT request_id, customer, zone FROM delivery_requests"):
        print(dict(row))

    print("\n--- Perishable requests only (WHERE clause) ---")
    cur.execute("SELECT * FROM delivery_requests WHERE package_type = ?", ("perishable",))
    for row in cur.fetchall():
        print(dict(row))

    print("\n--- Vehicles available in the OMR zone ---")
    cur.execute("SELECT * FROM fleet WHERE zone = ?", ("OMR",))
    for row in cur.fetchall():
        print(dict(row))

    print("\n--- Delivery requests joined with a matching vehicle in the same zone ---")
    cur.execute("""
        SELECT r.request_id, r.customer, r.zone, f.vehicle_id, f.vehicle_type
        FROM delivery_requests r
        JOIN fleet f ON f.zone = r.zone
        ORDER BY r.request_id
    """)
    for row in cur.fetchall():
        print(dict(row))

    print("\n--- UPDATE: bump VAN-04's ETA to 12 minutes ---")
    cur.execute("UPDATE fleet SET eta_minutes = ? WHERE vehicle_id = ?", (12, "VAN-04"))
    conn.commit()
    cur.execute("SELECT vehicle_id, eta_minutes FROM fleet WHERE vehicle_id = 'VAN-04'")
    print(dict(cur.fetchone()))

    print("\n--- DELETE: remove REQ-903 (delivered already) ---")
    cur.execute("DELETE FROM delivery_requests WHERE request_id = ?", ("REQ-903",))
    conn.commit()
    cur.execute("SELECT request_id FROM delivery_requests")
    print("Remaining requests:", [row["request_id"] for row in cur.fetchall()])

    print("\n--- assignments table (starts empty; sriroute_tools_db.assign_courier writes here) ---")
    cur.execute("SELECT * FROM assignments")
    print(cur.fetchall())

    conn.close()


if __name__ == "__main__":
    build_database()
    print(f"Created {DB_PATH}")
    demo_queries()
