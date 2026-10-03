"""SMT-based safety check for warehouse dispatch assignments."""

from __future__ import annotations

from typing import Any, Dict, List

import z3
from pydantic import BaseModel, Field


class PackageSpec(BaseModel):
    id: str
    weight: float = Field(gt=0)


class RobotSpec(BaseModel):
    id: str
    capacity: float = Field(gt=0)
    battery_pct: float = Field(ge=0, le=100)
    min_battery: float = Field(default=20, ge=0, le=100)
    drain_rate: float = Field(default=0, ge=0)


class DispatchPayload(BaseModel):
    packages: List[PackageSpec]
    robots: List[RobotSpec]
    requested_assignments: Dict[str, str] = Field(default_factory=dict)


def verify_dispatch_smt(raw_payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Hard safety gate for a proposed dispatch plan.

    The solver decides one robot index for every package and enforces:
      1. every package receives exactly one robot;
      2. robots below min_battery receive nothing;
      3. assigned weight <= robot capacity;
      4. battery_pct - drain_rate * assigned_weight >= min_battery.

    On SAT, assignments are returned.
    On UNSAT, diagnostics are returned for ReAct reflection.
    """
    try:
        payload = DispatchPayload.model_validate(raw_payload)
    except Exception as exc:
        return {
            "status": "SYNTACTIC_ERROR",
            "message": str(exc),
        }

    if not payload.packages:
        return {
            "status": "SYNTACTIC_ERROR",
            "message": "At least one package is required.",
        }

    if not payload.robots:
        return {
            "status": "SYNTACTIC_ERROR",
            "message": "At least one robot is required.",
        }

    solver = z3.Solver()
    robot_count = len(payload.robots)

    assign = {
        package.id: z3.Int(f"assign_{i}")
        for i, package in enumerate(payload.packages)
    }

    # Every package must map to exactly one valid robot index.
    for var in assign.values():
        solver.add(var >= 0, var < robot_count)

    # Treat explicit human assignments as hard constraints when they are supplied.
    robot_index_by_id = {robot.id: idx for idx, robot in enumerate(payload.robots)}
    for package_id, robot_id in payload.requested_assignments.items():
        if package_id not in assign:
            return {
                "status": "SYNTACTIC_ERROR",
                "message": f"Requested assignment references unknown package {package_id}.",
            }
        if robot_id not in robot_index_by_id:
            return {
                "status": "SYNTACTIC_ERROR",
                "message": f"Requested assignment references unknown robot {robot_id}.",
            }
        solver.add(assign[package_id] == robot_index_by_id[robot_id])

    # Enforce battery, capacity, and drain-rate constraints.
    for robot_idx, robot in enumerate(payload.robots):
        assigned_weight = z3.Sum([
            z3.If(assign[p.id] == robot_idx, p.weight, 0.0)
            for p in payload.packages
        ])

        if robot.battery_pct < robot.min_battery:
            for var in assign.values():
                solver.add(var != robot_idx)

        solver.add(assigned_weight <= robot.capacity)
        solver.add(
            robot.battery_pct
            - robot.drain_rate * assigned_weight
            >= robot.min_battery
        )

    result = solver.check()

    if result == z3.sat:
        model = solver.model()
        assignments = {
            p.id: payload.robots[model[assign[p.id]].as_long()].id
            for p in payload.packages
        }
        return {
            "status": "SATISFIABLE",
            "assignments": assignments,
            "feedback": "Dispatch plan verified by SMT.",
        }

    # Keep the failure details concrete enough for the LLM to reason about them.
    active = [
        r for r in payload.robots
        if r.battery_pct >= r.min_battery
    ]
    disabled = [
        r for r in payload.robots
        if r.battery_pct < r.min_battery
    ]

    diagnostics = {
        "total_requested_weight": sum(p.weight for p in payload.packages),
        "active_robots": [r.id for r in active],
        "disabled_robots_low_battery": [r.id for r in disabled],
        "largest_active_capacity": max(
            (r.capacity for r in active),
            default=0,
        ),
    }

    if not active:
        feedback = (
            "UNSAT: every robot is below its minimum battery threshold."
        )
    else:
        oversized = [
            p.id
            for p in payload.packages
            if p.weight > max(r.capacity for r in active)
        ]
        diagnostics["oversized_packages"] = oversized

        if oversized:
            feedback = (
                "UNSAT: at least one package exceeds the capacity of every "
                "currently active robot."
            )
        else:
            feedback = (
                "UNSAT: no assignment satisfies all active-robot capacity "
                "and battery constraints simultaneously."
            )

        drain_robots = [
            r.id for r in active if r.drain_rate > 0
        ]
        if drain_robots:
            diagnostics["robots_with_battery_drain"] = drain_robots
            feedback += (
                " Battery drain is active on at least one robot and may "
                "make an otherwise capacity-feasible assignment invalid."
            )

    return {
        "status": "UNSATISFIABLE",
        "diagnostics": diagnostics,
        "feedback": feedback,
    }
