"""ReAct-based dispatch agent with a final SMT safety gate."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

# Load a local .env file before the existing lab LLM client is instantiated.
# This is capstone integration code; the lab agent.py remains untouched.
load_dotenv()

from agent import AgentStep, LLMClient, SeenPayloads, parse_step
from pydantic import ValidationError

from capstone_smt_guard import verify_dispatch_smt


MAX_TURNS = 8


def explicit_item_robot_pairs(user_request: str) -> Dict[str, str]:
    """Extract explicit ITEM -> Robot mappings from the human request."""
    pairs: Dict[str, str] = {}
    pattern = re.compile(
        r"\b(ITEM-\d+)\b\s*(?:to|->|assigned\s+to)\s*"
        r"(?:Robot\s*)[-#:]*\s*(\d+)\b",
        flags=re.IGNORECASE,
    )
    for item_id, robot_id in pattern.findall(user_request):
        pairs[item_id.upper()] = str(int(robot_id))
    return pairs


SYSTEM_PROMPT = """
You are the dispatch reasoning layer of an autonomous warehouse.

Return exactly one JSON object per turn:
{
  "thought": "<brief reasoning summary>",
  "action": "<action>",
  "action_input": {...}
}

Allowed actions are exactly:

1. "verify_dispatch_smt"
   action_input:
   {
     "packages": [
       {"id": "<package id>", "weight": <number>}
     ],
     "robots": [
       {
         "id": "<robot id>",
         "capacity": <number>,
         "battery_pct": <number>,
         "min_battery": <number>,
         "drain_rate": <number>
       }
     ]
   }

2. "ask_user"
   action_input:
   {"question": "<question>"}

3. "finish"
   action_input:
   {
     "answer": "<answer>",
     "assignments": {"package_id": "robot_id", ...}
   }

Rules:
- First translate the warehouse request into a concrete verifier payload.
- Only create assignments for packages/items that the HUMAN DISPATCH REQUEST explicitly asks to move.
- Do not add extra packages, extra tasks, or "helpful" assignments that were not requested.
- If the human explicitly names a robot for a requested package, use that robot; do not silently substitute another robot.
- When the HUMAN DISPATCH REQUEST explicitly says "ITEM-X to Robot-Y", the
  final assignment for ITEM-X MUST be exactly Robot-Y. Do not move that item to
  another robot merely because SMT also finds that alternative feasible.
- Do not treat a merely feasible alternative assignment as permission to rewrite
  an explicit human assignment.
- The SMT verifier is the source of truth for physical feasibility, while the
  human request is the source of truth for explicit assignment intent.
- If SMT returns SATISFIABLE, finish using exactly the assignments from that
  verified result.
- If SMT returns UNSATISFIABLE, read diagnostics and feedback, explain the
  actual constraint failure in your Thought, and propose a genuinely
  different payload.
- Never submit the exact same verify_dispatch_smt payload twice.
- A changed plan must be sent back through SMT before being described as
  feasible.
- Use ask_user when the correct relaxation requires a human preference.
- Never claim a plan is feasible without a SATISFIABLE verification for that
  exact plan.
"""


class WarehouseReActAgent:
    def __init__(self, model: str = "gpt-5-mini"):
        self.llm = LLMClient(model=model)

    def run(
        self,
        user_request: str,
    ) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_request},
        ]

        required_pairs = explicit_item_robot_pairs(user_request)
        seen = SeenPayloads()
        last_sat_assignments = None

        for turn in range(1, MAX_TURNS + 1):
            raw = self.llm.complete(messages)
            messages.append({"role": "assistant", "content": raw})

            try:
                step: AgentStep = parse_step(raw)
            except ValidationError as exc:
                observation = {
                    "status": "SYNTACTIC_ERROR",
                    "message": (
                        "Return exactly one valid AgentStep JSON object. "
                        + str(exc)
                    ),
                }
                messages.append({
                    "role": "user",
                    "content": json.dumps(observation),
                })
                continue

            if step.action == "verify_dispatch_smt":
                payload = dict(step.action_input)

                # Keep any explicit human ITEM -> Robot intent as a hard SMT
                # constraint, rather than letting the solver pick a different robot.
                if required_pairs:
                    payload["requested_assignments"] = dict(required_pairs)

                if seen.seen(payload):
                    observation = {
                        "status": "UNSATISFIABLE",
                        "feedback": (
                            "This exact payload was already submitted. "
                            "Change the plan before verifying again."
                        ),
                    }
                else:
                    seen.add(payload)
                    observation = verify_dispatch_smt(payload)
                    if observation.get("status") == "SATISFIABLE":
                        sat_assignments = {
                            str(item_id).upper(): str(robot_id)
                            for item_id, robot_id in observation.get("assignments", {}).items()
                        }
                        normalized_required = {
                            str(item_id).upper(): str(robot_id)
                            for item_id, robot_id in required_pairs.items()
                        }
                        if normalized_required and any(
                            sat_assignments.get(item_id) != robot_id
                            for item_id, robot_id in normalized_required.items()
                        ):
                            observation = {
                                "status": "SCOPE_VIOLATION",
                                "feedback": (
                                    "SMT found a feasible assignment that conflicts with an explicit "
                                    "human robot assignment. SMT feasibility does not override explicit "
                                    "human intent. Rebuild the verifier payload so that the requested "
                                    "ITEM -> Robot mappings are preserved, then verify the changed payload."
                                ),
                                "requested_assignments": normalized_required,
                                "smt_assignments": sat_assignments,
                            }
                        else:
                            last_sat_assignments = observation["assignments"]

                messages.append({
                    "role": "user",
                    "content": json.dumps(observation),
                })
                continue

            if step.action == "ask_user":
                question = str(
                    step.action_input.get(
                        "question",
                        "Please choose how the dispatch should be relaxed.",
                    )
                )
                # For CLI use, ask the user in the terminal; the GUI can replace
                # this later if needed.
                try:
                    answer = input(question + "\n> ")
                except EOFError:
                    answer = "Use the least disruptive safe relaxation."

                messages.append({
                    "role": "user",
                    "content": json.dumps({
                        "status": "USER_REPLY",
                        "answer": answer,
                    }),
                })
                continue

            if step.action == "finish":
                assignments = step.action_input.get("assignments")

                if assignments is not None:
                    if last_sat_assignments is None:
                        messages.append({
                            "role": "user",
                            "content": json.dumps({
                                "status": "ERROR",
                                "message": (
                                    "A dispatch assignment cannot be "
                                    "presented as feasible before SMT "
                                    "returns SATISFIABLE."
                                ),
                            }),
                        })
                        continue

                    if assignments != last_sat_assignments:
                        messages.append({
                            "role": "user",
                            "content": json.dumps({
                                "status": "ERROR",
                                "message": (
                                    "Finish assignments must match the "
                                    "latest SATISFIABLE SMT result."
                                ),
                            }),
                        })
                        continue

                return step.action_input, messages

            messages.append({
                "role": "user",
                "content": json.dumps({
                    "status": "ERROR",
                    "message": (
                        "Unknown action. Choose verify_dispatch_smt, "
                        "ask_user, or finish."
                    ),
                }),
            })

        return {
            "answer": "Turn budget exhausted before a verified result.",
            "assignments": {},
        }, messages


def warehouse_context_from_state(state) -> Dict[str, Any]:
    """
    Convert the live WarehouseState into the language-model context.

    This function deliberately does not alter the existing state classes.
    It only reads the attributes already used by the orchestrator.
    """
    packages = []
    for item in state.inventory.values():
        packages.append({
            "id": str(item.item_id),
            "weight": float(item.weight),
        })

    robots = []
    for robot in state.robots.values():
        robots.append({
            "id": str(robot.robot_id),
            "capacity": float(robot.capacity),
            "battery_pct": float(robot.battery),
            "min_battery": 20.0,
            "drain_rate": 0.0,
        })

    return {
        "packages": packages,
        "robots": robots,
    }


def build_request_with_context(
    natural_language_request: str,
    state,
) -> str:
    context = warehouse_context_from_state(state)
    return (
        natural_language_request
        + "\n\nCURRENT WAREHOUSE STATE:\n"
        + json.dumps(context, indent=2)
    )


if __name__ == "__main__":
    request = (
        "Dispatch Item-1 and Item-2 using the available robots. "
        "Choose a safe feasible allocation."
    )
    result, trace = WarehouseReActAgent().run(request)
    print(json.dumps(result, indent=2))
