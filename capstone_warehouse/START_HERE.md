# Ishaan's Autonomous Warehouse Capstone

This project is based on and extended using the six laboratory projects:

- ReAct Lab -> LLM/ReAct command interpretation
- Lab (Updated) -> Pydantic + Z3/SMT verification/reflection
- MDP Lab -> MDP/value/policy planning concepts and GUI patterns
- Q-learning Lab -> Q-learning/SARSA adaptive control
- RL Prediction Lab -> TD/Monte-Carlo value estimation diagnostics
- Bandits Lab -> warehouse exploration/zone selection

## First demo target: 19/09/2026

Working pipeline:

```text
Human command
    -> ReAct/LLM
    -> structured assignment
    -> Z3 safety gate
    -> A* global route
    -> SARSA local controller
    -> HMM perception
    -> simulated robot
    -> live Tkinter dashboard
```

The conflict resolver and SQLite memory are already integrated in the reference architecture and remain part of the target system.

## Run on Windows

1. Create and activate a virtual environment.
2. Install the packages in `requirements.txt`.
3. Copy `.env.example` to `.env` and set `OPENAI_API_KEY`.
4. Run:

```powershell
python capstone_dashboard_multi_robot.py
```

Or:

```powershell
python capstone_launcher.py
```

The GUI accepts natural-language warehouse commands. The default command is designed to demonstrate an explicit ITEM -> Robot assignment.

## Demo commands

### Feasible

```text
Assign ITEM-001 to Robot 1 and deliver it to TASK-001's destination. Use the safest feasible plan.
```

### Verification failure / reflection

```text
Assign ITEM-001 to Robot 2 and deliver it to TASK-001's destination.
```

Use the second command only if the current warehouse state makes it fail; the point is to demonstrate that the deterministic guard controls execution.

## Important design rule

The LLM never directly moves a robot. It proposes a structured dispatch. Deterministic Python scope checks and Z3 verification run before a task is assigned, and the physical simulator executes only a verified assignment.
