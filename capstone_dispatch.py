"""Capstone dispatch layer.

This file reads the live warehouse state, asks the ReAct agent for a dispatch,
then validates the result before assigning an existing task to a robot.
The physical execution still lives in the existing orchestrator.
"""

from __future__ import annotations

import json
import re
from typing import Any, Dict

from capstone_react_agent import WarehouseReActAgent
from capstone_smt_guard import verify_dispatch_smt


def explicit_item_robot_pairs(user_request: str) -> Dict[str, str]:
    """Extract explicit ITEM -> Robot assignments from the human request."""
    pairs: Dict[str, str] = {}
    pattern = re.compile(
        r"\b(ITEM-\d+)\b\s*(?:to|->|assigned\s+to)\s*"
        r"(?:Robot\s*)[-#:]*\s*(\d+)\b",
        flags=re.IGNORECASE,
    )
    for item_id, robot_id in pattern.findall(user_request):
        pairs[item_id.upper()] = str(int(robot_id))
    return pairs


def warehouse_state_context(state) -> Dict[str, Any]:
    """Serialize the existing WarehouseState into LLM-readable context."""
    robots = []
    for robot in state.robots.values():
        robots.append(
            {
                "id": robot.robot_id,
                "capacity_kg": robot.capacity,
                "battery_pct": robot.battery,
                "wear": robot.wear,
                # The current warehouse model does not track a separate drain rate,
                # so the capstone layer keeps this at zero unless the model adds it.
                "min_battery_pct": 20.0,
                "drain_rate_pct_per_kg": 0.0,
            }
        )

    items = []
    for item in state.inventory.values():
        items.append(
            {
                "id": item.item_id,
                "weight_kg": item.weight,
                "location": item.location,
                "status": item.status,
            }
        )

    tasks = []
    for task in state.tasks.values():
        tasks.append(
            {
                "task_id": task.task_id,
                "item_id": task.item_id,
                "destination": task.destination,
                "assigned_robot_id": getattr(task, "assigned_robot_id", None),
            }
        )

    return {
        "robots": robots,
        "items": items,
        "tasks": tasks,
    }


def build_dispatch_request(user_request: str, state) -> str:
    """Combine the human command with authoritative warehouse state."""
    context = warehouse_state_context(state)

    return (
        "HUMAN DISPATCH REQUEST:\n"
        f"{user_request}\n\n"
        "AUTHORITATIVE CURRENT WAREHOUSE STATE:\n"
        f"{json.dumps(context, indent=2)}\n\n"
        "Use the supplied warehouse state for package weights, robot "
        "capacity, battery, and available entities. Do not ask the human "
        "to repeat information already present here."
    )


def enforce_human_request_scope(
    user_request: str,
    assignments: Dict[str, str],
) -> None:
    """Reject assignments that go beyond the human request or override an explicit robot mapping."""
    explicit_items = {m.upper() for m in re.findall(r"\bITEM-\d+\b", user_request, flags=re.IGNORECASE)}
    explicit_robots = {int(m) for m in re.findall(r"\brobot\s*[-#:]?\s*(\d+)\b", user_request, flags=re.IGNORECASE)}
    explicit_pairs = explicit_item_robot_pairs(user_request)

    normalized_assignments = {
        str(item_id).upper(): str(robot_id)
        for item_id, robot_id in assignments.items()
    }

    if explicit_items:
        extra_items = set(normalized_assignments) - explicit_items
        missing_items = explicit_items - set(normalized_assignments)
        if extra_items or missing_items:
            raise ValueError(
                "LLM assignment scope does not match the human request. "
                f"Extra items: {sorted(extra_items)}; missing items: {sorted(missing_items)}; "
                f"requested items: {sorted(explicit_items)}."
            )

    if explicit_pairs:
        mismatches = {
            item_id: (normalized_assignments.get(item_id), robot_id)
            for item_id, robot_id in explicit_pairs.items()
            if normalized_assignments.get(item_id) != robot_id
        }
        if mismatches:
            raise ValueError(
                "LLM assignment conflicts with explicit human ITEM -> Robot mapping: "
                f"{mismatches}. The explicit human mapping must be preserved."
            )

    if len(explicit_robots) == 1 and not explicit_pairs:
        requested_robot = next(iter(explicit_robots))
        wrong_robots = {str(robot_id) for robot_id in normalized_assignments.values() if str(robot_id) != str(requested_robot)}
        if wrong_robots:
            raise ValueError(
                "LLM attempted to assign a requested item to a different robot than "
                f"the one explicitly named by the human (Robot {requested_robot}): "
                f"{sorted(wrong_robots)}."
            )


def verify_assignment_against_state(
    assignments: Dict[str, str],
    state,
) -> None:
    """Final validation before we touch the live warehouse state."""
    # The SMT layer uses string IDs, while the warehouse model stores integers.
    # Normalize at this boundary instead of changing the underlying model files.
    known_items = {str(item.item_id) for item in state.inventory.values()}
    known_robots = {str(robot.robot_id) for robot in state.robots.values()}

    unknown_items = set(map(str, assignments)) - known_items
    unknown_robots = set(map(str, assignments.values())) - known_robots

    if unknown_items:
        raise ValueError(
            f"SMT returned unknown item IDs: {sorted(unknown_items)}"
        )

    if unknown_robots:
        raise ValueError(
            f"SMT returned unknown robot IDs: {sorted(unknown_robots)}"
        )


def dispatch_verified_plan(
    user_request: str,
    state,
    orchestrator,
) -> Dict[str, Any]:
    """Run the dispatch gate and return the verified assignment payload."""
    prompt = build_dispatch_request(user_request, state)

    agent = WarehouseReActAgent()
    result, trace = agent.run(prompt)

    assignments = result.get("assignments", {})
    enforce_human_request_scope(user_request, assignments)
    if not assignments:
        return {
            "status": "NO_DISPATCH",
            "result": result,
            "trace": trace,
        }

    verify_assignment_against_state(assignments, state)

    # Rebuild the payload from the verified assignment and check it again right
    # before execution, just in case the state changed in the meantime.
    robots_by_id = {str(r.robot_id): r for r in state.robots.values()}
    items_by_id = {str(i.item_id): i for i in state.inventory.values()}

    # Keep the SMT result in the string-ID form the capstone checks expect.
    assignments = {str(item_id): str(robot_id) for item_id, robot_id in assignments.items()}

    payload = {
        "packages": [
            {
                "id": item_id,
                "weight": float(items_by_id[item_id].weight),
            }
            for item_id in assignments
        ],
        "robots": [
            {
                "id": robot_id,
                "capacity": float(robot.capacity),
                "battery_pct": float(robot.battery),
                "min_battery": 20.0,
                "drain_rate": 0.0,
            }
            for robot_id in sorted(set(assignments.values()))
            for robot in [robots_by_id[robot_id]]
        ],
        "requested_assignments": dict(assignments),
    }

    final_check = verify_dispatch_smt(payload)
    if final_check.get("status") != "SATISFIABLE":
        raise RuntimeError(
            "Dispatch was verified during ReAct but failed the final "
            f"pre-execution SMT check: {final_check}"
        )

    # Only assign tasks after the final SMT check passes.
    created_or_assigned = []

    for item_id, robot_id in assignments.items():
        item = items_by_id[item_id]

        matching_tasks = [
            task
            for task in state.tasks.values()
            if task.item_id == item_id
        ]

        if not matching_tasks:
            raise ValueError(
                f"No existing Warehouse Task found for item {item_id}. "
                "Create the task from the application's task interface "
                "before executing a dispatch."
            )

        task = matching_tasks[0]

        # Convert the SMT string ID back to the live model's integer ID.
        live_robot_id = robots_by_id[robot_id].robot_id
        orchestrator.assign_task(
            task.task_id,
            live_robot_id,
        )

        created_or_assigned.append(
            {
                "task_id": task.task_id,
                "item_id": item_id,
                "robot_id": robot_id,
            }
        )

    return {
        "status": "DISPATCH_AUTHORIZED",
        "result": result,
        "final_smt_check": final_check,
        "task_assignments": created_or_assigned,
        "trace": trace,
    }
