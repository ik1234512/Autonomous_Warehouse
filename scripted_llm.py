"""
scripted_llm.py
==================
RECREATED (content matches the original file read earlier this session,
before it was deleted) -- a deterministic stand-in for a real LLM, used
by test_correctness.py so you can check your ReAct loop without needing
an API key or paying for real model calls.

A ScriptedLLM just plays back a fixed list of JSON responses, one per
call to .step(), regardless of what's actually in `messages` -- it does
not reason about the conversation at all. That's fine for exercising
your loop's control flow against a FIXED scenario, but remember these
scripts will not adapt if your loop's error messages differ from what's
expected here -- they test your loop, not your prompt design.

No SQLite-specific changes needed here: these scripts only emit
{"thought", "action", "action_input"} JSON -- they never touch the tool
backend directly, so the exact same scripts work whether react_loop_lab
is wired to sriroute_tools (in-memory) or sriroute_tools_db (SQLite).
"""

import json


class ScriptedLLM:
    def __init__(self, responses):
        self.responses = responses
        self.call_count = 0

    def step(self, messages):
        if self.call_count >= len(self.responses):
            raise RuntimeError(
                "ScriptedLLM ran out of scripted responses -- your loop called "
                "it more times than this scenario expects. Check you're not "
                "re-issuing the same call, or stuck in a loop that never "
                "reaches 'finish'."
            )
        resp = self.responses[self.call_count]
        self.call_count += 1
        return resp if isinstance(resp, str) else json.dumps(resp)


def happy_path_script():
    """REQ-901: standard package, South Chennai has vehicles -> assign and finish."""
    return ScriptedLLM([
        {"thought": "Look up the request first.", "action": "get_delivery_request",
         "action_input": {"request_id": "REQ-901"}},
        {"thought": "Check what's available in South Chennai.", "action": "check_fleet_availability",
         "action_input": {"zone": "South Chennai"}},
        {"thought": "VAN-04 has plenty of capacity for 4kg.", "action": "assign_courier",
         "action_input": {"request_id": "REQ-901", "vehicle_id": "VAN-04"}},
        {"thought": "Done.", "action": "finish",
         "action_input": {"final_answer": "REQ-901 assigned to VAN-04."}},
    ])


def constraint_gated_script():
    """REQ-902: perishable, but OMR has no refrigerated vehicle -> must
    report infeasible, not assign the bike anyway."""
    return ScriptedLLM([
        {"thought": "Look up the request.", "action": "get_delivery_request",
         "action_input": {"request_id": "REQ-902"}},
        {"thought": "It's perishable -- check OMR's fleet.", "action": "check_fleet_availability",
         "action_input": {"zone": "OMR"}},
        {"thought": "Only a bike is available and it isn't refrigerated -- "
                     "I cannot safely fulfill this request.",
         "action": "finish",
         "action_input": {"final_answer": "Cannot fulfill REQ-902: no refrigerated "
                                            "vehicle available in OMR for this perishable item."}},
    ])


def malformed_then_recover_script():
    """REQ-901 again, but the agent first sends a malformed assign_courier
    call (missing the required vehicle_id) and must recover after seeing
    the validation-error observation."""
    return ScriptedLLM([
        {"thought": "Look up the request.", "action": "get_delivery_request",
         "action_input": {"request_id": "REQ-901"}},
        {"thought": "Check South Chennai's fleet.", "action": "check_fleet_availability",
         "action_input": {"zone": "South Chennai"}},
        # Deliberately malformed: missing required "vehicle_id"
        {"thought": "Assign the van.", "action": "assign_courier",
         "action_input": {"request_id": "REQ-901"}},
        # Recovery, after seeing the validation-error observation:
        {"thought": "I need to include vehicle_id.", "action": "assign_courier",
         "action_input": {"request_id": "REQ-901", "vehicle_id": "VAN-04"}},
        {"thought": "Done.", "action": "finish",
         "action_input": {"final_answer": "REQ-901 assigned to VAN-04 (after correcting a malformed call)."}},
    ])


def surge_script(expect_approval: bool):
    """REQ-904: tight window, the standard van is too slow -> request
    surge approval. Pair this with an approval_fn in the test that
    actually returns `expect_approval`, so the script and the injected
    human decision agree."""
    final = ("REQ-904 assigned to PRIORITY-VAN-01 (surge approved)." if expect_approval
             else "REQ-904 could not meet its window; surge was denied, customer notified of delay.")
    steps = [
        {"thought": "Look up the request.", "action": "get_delivery_request",
         "action_input": {"request_id": "REQ-904"}},
        {"thought": "Check Sri City's fleet.", "action": "check_fleet_availability",
         "action_input": {"zone": "Sri City"}},
        {"thought": "Standard VAN-09's ETA is 25 min but the window closes in 15 -- "
                     "only the priority van can make it, and that costs extra. Ask for approval.",
         "action": "request_surge_approval",
         "action_input": {"request_id": "REQ-904", "extra_cost": 450.0,
                           "reason": "Only the priority van can make the 15-minute window."}},
    ]
    if expect_approval:
        steps.append({"thought": "Approved -- assign the priority van.", "action": "assign_courier",
                      "action_input": {"request_id": "REQ-904", "vehicle_id": "PRIORITY-VAN-01"}})
    steps.append({"thought": "Done.", "action": "finish", "action_input": {"final_answer": final}})
    return ScriptedLLM(steps)
