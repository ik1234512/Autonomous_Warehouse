"""
llm_client.py
====================
RECONSTRUCTED, not a byte-exact restore. This file was deleted earlier in
the session before its contents were ever read, so unlike the other four
recreated files, this one is rebuilt from its known interface rather than
copied: react_loop_solution.py does `from llm_client import OpenAILLM`
and calls `.step(messages) -> str` on it, and LAB_HANDOUT.md describes it
as "a live call to a model instead of a scripted script." The .env file
holds a key in the `sk-proj-...` format, which is OpenAI's, so this
version calls the OpenAI API rather than Anthropic's. If your original
file did something different (different model provider, a different
system prompt, retry/backoff logic, etc.), this won't match it --
treat it as a working stand-in, not a recovered original.

Wraps a live OpenAI chat-completions call behind the same `.step(messages)`
interface as ScriptedLLM (scripted_llm.py), so it's a drop-in swap for
run_react_loop(). The system prompt describes the SQLite-backed tool set
from sriroute_tools_db.py -- same tool names and argument shapes as
sriroute_tools.py, since both share TOOL_ARG_MODELS.

Setup:
    pip install openai python-dotenv
    Make sure .env contains a line of the form:
        OPENAI_API_KEY=sk-...
    (a bare key with no "OPENAI_API_KEY=" prefix will NOT be picked up
    by load_dotenv() -- fix the .env file's format if needed.)
"""

import json
import os

from dotenv import load_dotenv
from openai import OpenAI

from sriroute_tools import TOOL_ARG_MODELS

load_dotenv()

_TOOL_ARGS_SUMMARY = json.dumps(
    {name: list(model.model_fields.keys()) for name, model in TOOL_ARG_MODELS.items()},
    indent=2,
)

SYSTEM_PROMPT = f"""You are a dispatch agent for SriRoute, a delivery service.

Given a natural-language dispatch request, decide whether to assign a
courier immediately, escalate for human approval via surge pricing, or
report the request as infeasible -- using only the tool observations you
receive, one call at a time.

At every turn, respond with ONLY a single JSON object, no other text:
{{"thought": "<your reasoning>", "action": "<tool name or 'finish'>", "action_input": {{...}}}}

Available actions and their required argument fields:
{_TOOL_ARGS_SUMMARY}

Call "finish" with {{"final_answer": "..."}} once you have a final answer.
Do not fabricate tool results -- only act on what an Observation message
actually told you.
"""


class OpenAILLM:
    def __init__(self, model: str = "gpt-5-mini", client: OpenAI | None = None):
        self.model = model
        self.client = client or OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    def step(self, messages: list[dict]) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}] + messages,
        )
        return response.choices[0].message.content
