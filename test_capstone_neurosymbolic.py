"""
Local, API-free tests for the capstone SMT safety layer.

Run:
    python test_capstone_neurosymbolic.py

This does not touch any lab file.
"""

from capstone_smt_guard import verify_dispatch_smt


def test_sat() -> None:
    payload = {
        "packages": [
            {"id": "Item-1", "weight": 6},
            {"id": "Item-2", "weight": 8},
        ],
        "robots": [
            {
                "id": "Robot-1",
                "capacity": 10,
                "battery_pct": 90,
                "min_battery": 20,
                "drain_rate": 0,
            },
            {
                "id": "Robot-2",
                "capacity": 20,
                "battery_pct": 90,
                "min_battery": 20,
                "drain_rate": 0,
            },
        ],
    }

    result = verify_dispatch_smt(payload)
    assert result["status"] == "SATISFIABLE", result
    assert set(result["assignments"]) == {"Item-1", "Item-2"}


def test_capacity_unsat() -> None:
    payload = {
        "packages": [
            {"id": "Heavy", "weight": 15},
        ],
        "robots": [
            {
                "id": "Robot-1",
                "capacity": 10,
                "battery_pct": 90,
                "min_battery": 20,
            },
            {
                "id": "Robot-2",
                "capacity": 12,
                "battery_pct": 90,
                "min_battery": 20,
            },
        ],
    }

    result = verify_dispatch_smt(payload)
    assert result["status"] == "UNSATISFIABLE", result
    assert "Heavy" in result["diagnostics"]["oversized_packages"]


def test_battery_gate_unsat() -> None:
    payload = {
        "packages": [
            {"id": "Item", "weight": 2},
        ],
        "robots": [
            {
                "id": "Robot-1",
                "capacity": 20,
                "battery_pct": 10,
                "min_battery": 20,
            },
        ],
    }

    result = verify_dispatch_smt(payload)
    assert result["status"] == "UNSATISFIABLE", result


def test_battery_drain_unsat() -> None:
    payload = {
        "packages": [
            {"id": "Cargo", "weight": 18},
        ],
        "robots": [
            {
                "id": "Robot-X",
                "capacity": 25,
                "battery_pct": 35,
                "min_battery": 20,
                "drain_rate": 1.0,
            },
            {
                "id": "Robot-Y",
                "capacity": 25,
                "battery_pct": 90,
                "min_battery": 20,
                "drain_rate": 0.3,
            },
        ],
    }

    result = verify_dispatch_smt(payload)
    assert result["status"] == "SATISFIABLE", result
    assert result["assignments"]["Cargo"] == "Robot-Y"


if __name__ == "__main__":
    test_sat()
    test_capacity_unsat()
    test_battery_gate_unsat()
    test_battery_drain_unsat()
    print("All capstone SMT guard tests passed.")
