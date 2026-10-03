"""
agent.py -- YOUR implementation of the manual ReAct execution loop for
the SMT Reflection Loop lab. See README.md for the full spec, and
cheatsheet-react-smt-z3.md for the ReAct loop shape and Z3 syntax.

Rename this file to agent.py before you start (that's the filename the
submission checklist expects) and fill in every TODO / NotImplementedError.

Rules:
  - No LangChain / LlamaIndex / any agent framework -- raw SDK calls,
    Pydantic validation, and a for loop only.
  - Do not edit verifier.py.

Usage (once implemented):
    cp .env.example .env       # then put your real OPENAI_API_KEY in .env -- see env-setup.md
    python agent.py sat        # Test Case 1 (feasible)
    python agent.py unsat      # Test Case 2 (over-constrained -> relaxation)
    python agent.py drain      # Test Case 3 (battery-drain -> relaxation, Task 4)
"""

from __future__ import annotations

import json
import sys
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

from verifier import verify_dispatch_smt

load_dotenv()  # reads OPENAI_API_KEY (etc.) from .env into os.environ -- see env-setup.md

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MAX_TURNS = 8            # Task 2: hard turn budget -- the loop must not run forever
DEFAULT_MODEL = "gpt-5-mini"


# ---------------------------------------------------------------------------
# Given: the step schema every LLM turn must validate against (Task 2)
# ---------------------------------------------------------------------------

class AgentStep(BaseModel):
    thought: str
    action: str
    action_input: Dict[str, Any]


# ---------------------------------------------------------------------------
# Given: the three graded evaluation prompts (README.md, Section 6)
# ---------------------------------------------------------------------------

TEST_CASES = {
    "sat": (
        "Allocate Item-1 (10kg) and Item-2 (15kg) across AGV-A (Capacity "
        "30kg, Battery 80%) and AGV-B (Capacity 20kg, Battery 50%). "
        "Minimum battery threshold is 25%."
    ),
    "unsat": (
        "Allocate Heavy-Box-1 (40kg) and Heavy-Box-2 (25kg) across AGV-1 "
        "(Capacity 30kg, Battery 90%) and AGV-2 (Capacity 20kg, Battery "
        "10%). Minimum battery threshold is 25%."
    ),
    "drain": (
        "Allocate Cargo-1 (22kg) and Cargo-2 (12kg) across AGV-X (Capacity "
        "25kg, Battery 35%, drains 1.0% battery per kg carried) and AGV-Y "
        "(Capacity 25kg, Battery 90%, drains 0.3% battery per kg carried). "
        "Minimum battery threshold is 20%."
    ),
}


# ---------------------------------------------------------------------------
# TODO (Task 1): system prompt
#
# Write the system prompt here. It must, at minimum:
#   - mandate exactly one JSON object per turn, matching AgentStep, and
#     nothing else (no prose outside the JSON, no markdown fences)
#   - define exactly three actions and their action_input shape:
#       1. verify_dispatch_smt  -> {"packages": [...], "agvs": [...]}
#          each AGV may also carry a "drain_rate" (percentage points of
#          battery per kg carried, default 0) -- only the "drain" test
#          case (Task 4) states one in the request; parse it into the
#          payload the same way you already parse "minimum battery
#          threshold" into min_battery.
#       2. ask_user             -> {"question": "<str>"}
#       3. finish                -> {"answer": "<str>", "assignments": {...}}
#   - state Key Rule 1: after an UNSATISFIABLE observation, the agent is
#     forbidden from resubmitting the exact same action_input -- it must
#     reflect on the diagnostics and either propose a genuinely different
#     (relaxed) payload, or call ask_user if the right relaxation depends
#     on a user preference
#   - state Key Rule 2: never call finish claiming feasibility unless the
#     most recent verify_dispatch_smt observation for that exact plan was
#     SATISFIABLE
#
# See README.md Section 5 (Task 1) for the full requirements, and
# cheatsheet-react-smt-z3.md Section 1 for the loop shape this prompt
# needs to fit into.
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """
You are a warehouse dispatch planning agent. Return exactly one JSON object on
every turn. Return no markdown, no code fences, and no text outside the JSON.
The JSON object must have exactly these fields:
{"thought": "...", "action": "...", "action_input": {...}}

The action must be exactly one of these three values:
1. verify_dispatch_smt
    action_input must contain packages and agvs. Each package has id and a
    positive integer weight. Each AGV has id, positive integer capacity,
    battery_pct, min_battery, and optional drain_rate. Parse the stated minimum
    battery threshold into min_battery. Parse phrases such as "drains 1.0%
    battery per kg carried" into drain_rate as a number.
2. ask_user
    action_input must be {"question": "a clear question"}. Use this when the
    correct relaxation requires a preference from the human.
3. finish
    action_input must contain an answer string and may contain assignments.

First parse the user's request into a verify_dispatch_smt action. Treat the
verifier observation as the only source of truth. If it returns SATISFIABLE,
call finish on the exact verified plan. If it returns UNSATISFIABLE, read the
diagnostics and feedback, write a new Thought that names the actual reason,
and either submit a concretely different relaxed payload or call ask_user.
Never submit the exact same action_input after UNSATISFIABLE. A relaxed plan
must be submitted to verify_dispatch_smt again before it is presented as
feasible. Never call finish with a feasibility claim unless the most recent
verification of that exact plan returned SATISFIABLE. If no relaxation works,
finish with an honest impossibility explanation that cites the diagnostics.
"""


# ---------------------------------------------------------------------------
# TODO (Task 2): JSON / plumbing helpers
# ---------------------------------------------------------------------------

def strip_fences(text: str) -> str:
    """Strip ```json ... ``` fences and any prose wrapped around the JSON
    object the model returned, so json.loads() gets a clean object.
    See cheatsheet-react-smt-z3.md Section 2 for the pattern."""
    cleaned = text.strip()
    if cleaned.startswith("```"):
        first_line_end = cleaned.find("\n")
        if first_line_end >= 0:
            cleaned = cleaned[first_line_end + 1:]
    if cleaned.endswith("```"):
        cleaned = cleaned[:-3].strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start >= 0 and end >= start:
        cleaned = cleaned[start:end + 1]
    return cleaned


def freeze(action_input: Dict[str, Any]) -> str:
    """Canonical string key for a JSON-able dict -- used by SeenPayloads
    below to detect a repeated verify_dispatch_smt payload. Dicts aren't
    hashable, but json.dumps(d, sort_keys=True) is."""
    return json.dumps(action_input, sort_keys=True)


class SeenPayloads:
    """Provided: tracks every action_input you've already sent to
    verify_dispatch_smt, so your run() loop (Task 2) can short-circuit a
    resubmitted UNSAT payload in code, instead of relying on the prompt
    alone (README Section 5, Task 2's "Repeated-payload detection" bullet).
    You don't need to write this class -- just call `.seen(...)` before
    calling verify_dispatch_smt, and `.add(...)` right after, inside run()."""

    def __init__(self) -> None:
        self._seen: set[str] = set()

    def seen(self, action_input: Dict[str, Any]) -> bool:
        return freeze(action_input) in self._seen

    def add(self, action_input: Dict[str, Any]) -> None:
        self._seen.add(freeze(action_input))


def parse_step(raw: str) -> AgentStep:
    """Parse + validate one model turn against AgentStep. Let
    pydantic.ValidationError propagate -- run() below handles it as the
    syntactic loop."""
    cleaned = strip_fences(raw)
    return AgentStep.model_validate_json(cleaned)


# ---------------------------------------------------------------------------
# TODO (Task 2): the LLM client
#
# A thin wrapper around a single raw API call. Keep it minimal -- this is
# NOT the place to add retries, agent frameworks, or extra abstraction.
# ---------------------------------------------------------------------------

class LLMClient:
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = model
        self.client = OpenAI()

    def complete(self, messages: List[Dict[str, str]]) -> str:
        """Send the full message history, return the raw text of the
        model's reply for this turn (before any parsing/stripping)."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        return response.choices[0].message.content


# ---------------------------------------------------------------------------
# TODO: ask_user action handler
# ---------------------------------------------------------------------------

def ask_user(question: str) -> str:
    """Prompt the human for a decision the agent can't make on its own.
    Non-interactive runs (e.g. automated grading) should fall back to a
    sensible default instead of blocking forever on stdin -- check
    sys.stdin.isatty()."""
    if sys.stdin.isatty():
        return input(question + "\n> ")
    return "Use the least disruptive relaxation and continue by verifying it."


# ---------------------------------------------------------------------------
# TODO (Task 2 + Task 3): the ReAct loop
#
# Must implement:
#   - message history that grows every turn (system, user, assistant,
#     user, assistant, ... -- observations go back in as "user" turns)
#   - the SYNTACTIC loop: catch pydantic.ValidationError from parse_step,
#     feed the validation error back to the model, retry -- never crash
#     the program on bad model JSON
#   - the LOGICAL loop: on an UNSATISFIABLE observation, feed the
#     `diagnostics` and `feedback` fields back verbatim; if the model
#     resubmits an action_input you've already seen for
#     verify_dispatch_smt, intercept it in code -- use the provided
#     SeenPayloads (call `.seen(...)` before verifying, `.add(...)` after)
#     rather than calling Z3 again or trusting the prompt alone
#   - a MAX_TURNS hard stop that raises/returns cleanly instead of
#     looping silently forever
#   - routing for all three actions: verify_dispatch_smt, ask_user, finish
#
# Return value: (result, messages) where `result` is the action_input
# passed to `finish`, and `messages` is the full trace for save_trace().
# ---------------------------------------------------------------------------

def run(llm, user_request: str) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    messages = []
    messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": user_request})

    seen_payloads = SeenPayloads()
    last_sat_payload = None
    last_sat_assignments = None
    turn = 0

    while turn < MAX_TURNS:
        turn = turn + 1
        raw = llm.complete(messages)
        messages.append({"role": "assistant", "content": raw})

        try:
            step = parse_step(raw)
        except ValidationError as error:
            observation = {
                "status": "SYNTACTIC_ERROR",
                "message": "Return one JSON object matching AgentStep. " + str(error),
            }
            messages.append({"role": "user", "content": json.dumps(observation)})
            continue
        except Exception as error:
            observation = {
                "status": "SYNTACTIC_ERROR",
                "message": "Return valid JSON. " + str(error),
            }
            messages.append({"role": "user", "content": json.dumps(observation)})
            continue

        if step.action == "verify_dispatch_smt":
            payload = step.action_input
            if seen_payloads.seen(payload):
                observation = {
                    "status": "UNSATISFIABLE",
                    "feedback": "This exact payload was already submitted. "
                    "You must propose a different payload and write a new Thought.",
                }
            else:
                seen_payloads.add(payload)
                observation = verify_dispatch_smt(payload)
                if observation.get("status") == "SATISFIABLE":
                    last_sat_payload = freeze(payload)
                    last_sat_assignments = observation.get("assignments")
            messages.append({"role": "user", "content": json.dumps(observation)})
            continue

        if step.action == "ask_user":
            question = step.action_input.get("question", "Please choose a relaxation.")
            answer = ask_user(str(question))
            observation = {"status": "USER_REPLY", "answer": answer}
            messages.append({"role": "user", "content": json.dumps(observation)})
            continue

        if step.action == "finish":
            answer = str(step.action_input.get("answer", ""))
            assignments = step.action_input.get("assignments")
            claims_feasible = False
            lower_answer = answer.lower()
            if assignments is not None:
                claims_feasible = True
            if "satisf" in lower_answer or "feasible" in lower_answer:
                claims_feasible = True

            if claims_feasible:
                if last_sat_payload is None:
                    observation = {
                        "status": "ERROR",
                        "message": "Verify a plan with SATISFIABLE before claiming feasibility.",
                    }
                    messages.append({"role": "user", "content": json.dumps(observation)})
                    continue
                if assignments is not None and assignments != last_sat_assignments:
                    observation = {
                        "status": "ERROR",
                        "message": "The finish assignments do not match the latest SATISFIABLE plan.",
                    }
                    messages.append({"role": "user", "content": json.dumps(observation)})
                    continue
            return step.action_input, messages

        observation = {
            "status": "ERROR",
            "message": "Unknown action. Choose verify_dispatch_smt, ask_user, or finish.",
        }
        messages.append({"role": "user", "content": json.dumps(observation)})

    result = {
        "answer": "The turn budget was exhausted before a verified result was reached.",
        "assignments": {},
    }
    messages.append({"role": "user", "content": json.dumps(result)})
    return result, messages


def save_trace(path: str, messages: List[Dict[str, str]]) -> None:
    with open(path, "w") as f:
        json.dump(messages, f, indent=2)
    print(f"Saved trace to {path}")


def main() -> None:
    args = sys.argv[1:]
    case = args[0] if args else "sat"
    if case not in TEST_CASES:
        raise SystemExit(f"Unknown test case '{case}', expected one of {list(TEST_CASES)}")

    user_request = TEST_CASES[case]

    llm = LLMClient()
    trace_path = f"trace_{case}.json"

    result, messages = run(llm, user_request)
    save_trace(trace_path, messages)

    print("\nFinal result:")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
