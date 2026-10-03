"""
react_loop_lab.py
====================
YOUR TASK: implement the ReAct dispatch loop.

RECREATED and adapted to the SQLite backend: this is the same exercise
as the original lab (see LAB_HANDOUT.md), but the tool registry now
comes from sriroute_tools_db.py -- so once you finish the TODOs below,
every assign_courier call your loop makes will write a durable row into
sriroute_fake.db's `assignments` table instead of just returning a
message and forgetting. The Pydantic schemas (TOOL_ARG_MODELS) are
unchanged -- they still come from sriroute_tools.py, which both the
in-memory and SQLite-backed tool sets share.

The LLM (or, for testing, the ScriptedLLM in scripted_llm.py) responds at
each turn with a JSON object:

    {"thought": "...", "action": "<tool name or 'finish'>", "action_input": {...}}

Each turn, your loop must:
  1. Parse that JSON. If it doesn't parse (json.JSONDecodeError), feed
     back an observation saying so, and move on to the next iteration --
     do NOT crash.
  2. Check the action name is a known tool or "finish" (see
     TOOL_ARG_MODELS in sriroute_tools.py). If not, feed back an
     observation listing the valid action names, and move on.
  3. Validate action_input against that tool's Pydantic argument model.
     If validation fails (pydantic.ValidationError), feed back the
     validation error message as the observation, and move on -- do NOT
     call the tool with invalid arguments.
  4. If action == "finish": validate action_input against FinishArgs and
     return (final_answer, messages) -- you're done.
     Otherwise: call the tool with the validated arguments, and append
     the tool's return value as the next observation message.

Run test_correctness.py to check your implementation against the four
scripted scenarios before doing anything else. It calls fake_db.build_
database() first, so sriroute_fake.db is rebuilt fresh every run.
"""

import json
from pydantic import ValidationError
from sriroute_tools import TOOL_ARG_MODELS
from sriroute_tools_db import make_tool_registry

MAX_ITERATIONS = 8


def run_react_loop(llm, request_text: str, approval_fn=None,
                    max_iterations=MAX_ITERATIONS, verbose=True):
    """
    llm: an object with a .step(messages: list[dict]) -> str method that
         returns the model's next raw text response (see scripted_llm.py
         and llm_client.py for two implementations of this interface).
    request_text: the natural-language dispatch request from the user.
    approval_fn: optional override for the human-approval step inside
                 request_surge_approval (see sriroute_tools_db.make_tool_registry).
                 Leave as None to use the real input()-based console prompt.

    Returns:
        (final_answer: str, transcript: list[dict])

    transcript is the full list of {"role": ..., "content": ...} messages
    accumulated over the loop -- test_correctness.py inspects this to
    check your loop took the right actions, not just that it produced a
    plausible-looking final answer.
    """
    tools = make_tool_registry(approval_fn)
    messages = [{"role": "user", "content": request_text}]

    for iteration in range(max_iterations):
        raw_response = llm.step(messages)
        messages.append({"role": "assistant", "content": raw_response})

        # TODO 1: parse raw_response as JSON (json.loads). On
        # json.JSONDecodeError, append an observation message (role
        # "user") explaining the response must be valid JSON matching
        # the required schema, and `continue`.
        try:
            step = json.loads(raw_response)
        except json.JSONDecodeError as e:
            observation = "Error: response must be valid JSON matching the required schema. Details: " + str(e)
            messages.append({"role": "user", "content": observation})
            continue

        # TODO 2: check step["action"] is a key in TOOL_ARG_MODELS (this
        # dict already includes "finish"). If not, append an observation
        # listing the valid action names (list(TOOL_ARG_MODELS.keys())),
        # and `continue`.
        if "action" not in step or step["action"] not in TOOL_ARG_MODELS:
            valid_actions = list(TOOL_ARG_MODELS.keys())
            observation = "Error: action must be one of " + str(valid_actions)
            messages.append({"role": "user", "content": observation})
            continue

        # TODO 3: look up model_cls = TOOL_ARG_MODELS[step["action"]] and
        # validate step.get("action_input", {}) against it:
        #     validated = model_cls(**action_input)
        # On pydantic.ValidationError, append str(e) as the observation
        # and `continue` -- do NOT proceed to call anything.
        model_cls = TOOL_ARG_MODELS[step["action"]]
        action_input = step.get("action_input", {})
        try:
            validated = model_cls(**action_input)
        except ValidationError as e:
            observation = str(e)
            messages.append({"role": "user", "content": observation})
            continue

        # TODO 4: if step["action"] == "finish": return
        # (validated.final_answer, messages).
        if step["action"] == "finish":
            return (validated.final_answer, messages)

        # TODO 5: otherwise, call tools[step["action"]](**validated.model_dump()),
        # append the JSON-stringified result as the next observation
        # message (role "user", content something like f"Observation: {json.dumps(result)}"),
        # and continue the loop.
        result = tools[step["action"]](**validated.model_dump())
        observation = "Observation: " + json.dumps(result)
        messages.append({"role": "user", "content": observation})

    raise RuntimeError(f"Exceeded max_iterations={max_iterations} without calling finish.")


if __name__ == "__main__":
    from fake_db import build_database
    from scripted_llm import happy_path_script

    build_database()
    final_answer, transcript = run_react_loop(happy_path_script(), "Fulfill pending order REQ-901.")
    print("Final answer:", final_answer)
