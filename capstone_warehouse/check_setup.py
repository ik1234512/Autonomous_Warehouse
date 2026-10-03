"""Check local prerequisites without importing optional runtime modules."""
from __future__ import annotations
import importlib.util
import os
import sys

REQUIRED = ["z3", "pydantic", "openai", "dotenv"]

print(f"Python: {sys.version.split()[0]}")
print(f"OPENAI_API_KEY set: {bool(os.getenv('OPENAI_API_KEY'))}")
print("Dependencies:")
missing = []
for name in REQUIRED:
    ok = importlib.util.find_spec(name) is not None
    print(f"  {'OK' if ok else 'MISSING'}  {name}")
    if not ok:
        missing.append(name)

print("Tkinter:", "OK" if importlib.util.find_spec("tkinter") else "MISSING")
if missing:
    print("\nInstall the packages listed in requirements.txt before running the dashboard.")
    raise SystemExit(1)
print("\nSetup looks ready.")
