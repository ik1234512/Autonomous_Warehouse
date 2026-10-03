# Smart Autonomous Warehouse

![Recording demo](Recording%202026-10-02%20175940.gif)

## 1. Project Overview

This project is an Autonomous Warehouse project. It models a multi-robot warehouse in software and combines a natural-language human interface with formal dispatch verification, probabilistic perception, classical route planning, conflict handling, adaptive movement, persistent memory, and a live dashboard.

The primary integrated demonstration is the Tkinter application in `capstone_dashboard_multi_robot.py`. A human enters a warehouse request, the application interprets it through a ReAct/LLM agent, validates and verifies the proposed assignment, and only then allows the existing warehouse execution components to move simulated robots. The implementation reuses existing warehouse and laboratory components where appropriate.

## 2. What the System Demonstrates

| Rubric layer | Repository implementation |
|---|---|
| Human interface and safety filter | Tkinter command entry, `WarehouseReActAgent`, Pydantic `AgentStep`, and the Z3-based `verify_dispatch_smt()` gate |
| Perception | `RobotPerception`, `HMMSensor`, and `HMMFilter` update estimated position from noisy observations |
| Route finder and conflict resolver | `ForkliftPlanner.a_star()` plans grid routes; `ConflictResolver` uses depth-2 alpha-beta Minimax for immediate collision or swap risks |
| Adaptive pilot | `WarehouseSARSA` is the existing on-policy local controller; `WarehouseQLearning` is an optional off-policy comparison controller. Both are constrained by the planned route |
| Stochastic planning and exploration | `WarehouseMDP` solves a noisy grid MDP with value iteration; `UCB1Bandit` is a new capstone UCB1 implementation and can select between A* and value iteration based on route-length reward |
| Memory | `WarehouseMemory` stores SQLite runs, events, robot snapshots, task snapshots, and item snapshots |
| Live dashboard | Tkinter renders the map, robots, estimates, routes, task state, events, and explicit ReAct trace fields |

## 3. System Architecture

```text
Human command in Tkinter
          |
          v
WarehouseReActAgent.run()
  OpenAI chat completion + Pydantic AgentStep
          |
          v
verify_dispatch_smt()  <---- authoritative WarehouseState
      | SAT                         | UNSAT / invalid
      v                             v
verified assignment queues     feedback, reconsideration,
      |                         or rejected dispatch
      v
WarehouseOrchestrator
      |
      +--> ForkliftPlanner.a_star() ----> route / reservations
      |
      +--> WarehouseSARSA -------------> local action proposal
      |
      +--> ConflictResolver ------------> conditional collision handling
      |
      v
Robot movement, pickup, delivery, HMM update
      |
      +--> WarehouseMemory / SQLite events and snapshots
      +--> Tkinter polling and live rendering
```

## 4. Human Command and Safety Pipeline

The dashboard stores the human request in `human_command_var` and starts execution from the **Run Command** control. `start_demo()` requires `OPENAI_API_KEY`, loads `.env` configuration, and runs the capstone pipeline in a worker thread so the Tkinter event loop can continue updating the interface.

`capstone_dispatch.py` builds an LLM prompt from the human request and the authoritative warehouse state. `WarehouseReActAgent` uses `openai.OpenAI().chat.completions.create(...)`; the default model configured by the capstone agent is `gpt-5-mini`. Each returned step is parsed as the Pydantic `AgentStep` model reused from `agent.py`. The capstone ReAct action set includes `verify_dispatch_smt`, `ask_user`, and `finish`, with a maximum of eight turns.

The application records explicit trace fields such as thought, action, observation, and verification result. These are displayed as an AI/ReAct trace; this documentation does not describe hidden chain-of-thought.

After a proposed assignment is returned, deterministic dispatch code enforces the human request scope. Explicit forms such as `ITEM-001 to Robot 1` are preserved. Unknown item or robot identifiers are rejected. A final SMT check is performed immediately before tasks are assigned, so dispatch authorization is separate from physical execution.

## 5. Perception: HMM

The simulated robot has a true warehouse position, but its sensor observations are noisy. `RobotPerception` uses the HMM components in `hmm_environment.py` and `hmm_filter.py` to estimate the hidden position. The sensor model is 80% accurate. The transition model gives an intended movement an 80% probability and models lateral slips at 10% each.

Warehouse positions use `(x, y)` coordinates, while the HMM uses `(row, column)` coordinates. The perception layer explicitly converts between these conventions. During movement, the estimate is updated from the simulated observation and the dashboard reports both true and estimated robot positions, making estimation error visible.

## 6. Navigation and Conflict Resolution

`ForkliftPlanner.a_star()` plans four-direction grid movement with unit movement cost and a Manhattan heuristic. Grid obstacles are excluded from candidate movement. During capstone execution, protected delivery goals and reserved routes are also considered when selecting a usable route. The resulting A* route is the global movement authority.

SARSA proposes a local action for the next leg, but `_execute_targeted_sarsa_move()` requires the proposed next position to remain on the current A* route and checks protected goals and reserved paths before moving. This separates global route planning from adaptive local control.

`ConflictResolver` in `conflict_resolver.py` implements depth-2 Minimax with alpha-beta pruning. Its evaluation penalizes same-cell collisions, position swaps, proximity, and waiting. The capstone invokes it for immediate collision or swap risk rather than claiming that every run necessarily exercises conflict resolution. A resolver outcome determines which movement option is allowed or preferred for that local conflict.

## 7. Adaptive Control: SARSA

`WarehouseSARSA` is trained with `WarehouseRLEnv`. Its action set contains north, south, east, west, and wait. In the integrated dashboard path, the controller is trained for 2,000 episodes for each immediate A* waypoint, then used to propose the next local action. The route, obstacles, protected goals, and reserved paths still constrain execution, so SARSA does not replace A*; it acts as the adaptive/local movement controller beneath the planned route.

## 7A. MDP Value Iteration and Bandit Route Selection

The uploaded capstone ZIP did not contain the original MDP lab source files, so `WarehouseMDP` is a new capstone-side implementation of the value-iteration algorithm rather than a direct file port. It models each traversable grid cell as a state and four cardinal moves as actions. The intended move occurs with probability 0.8; the two perpendicular slip outcomes share the remaining probability. Invalid moves leave the robot in place with a penalty. Value iteration computes a policy toward the requested goal. Extracted routes are checked for legal adjacent moves, and the orchestrator falls back to A* if policy extraction cannot produce a loop-free route. A* remains the default.

Set `WAREHOUSE_PLANNER=value_iteration` to use the MDP planner, or `WAREHOUSE_PLANNER=bandit` to use UCB1 to select between A* and value iteration. The bandit receives a simple route-efficiency reward, so this is an experimental strategy selector, not a collision or feasibility authority. The existing dispatch SMT guard and movement safety checks remain in place.

## 7B. Q-learning Navigation Comparison

`qlearning_lab.py` in the supplied repository is a gripper task with `(item_type, wear)` states and grasp actions. It is not a navigation learner. To compare on-policy SARSA with off-policy Q-learning fairly, `warehouse_qlearning.py` implements Q-learning against the same `WarehouseRLEnv` used by `WarehouseSARSA`. The orchestrator exposes `train_qlearning_for_target()` and `execute_qlearning_move()`. Execution accepts only the immediate next cell on the already planned route.

## 8. Memory: SQLite

The integrated dashboard creates `WarehouseMemory("warehouse_memory.db")`. `capstone_memory.py` defines the SQLite tables `runs`, `events`, `robot_snapshots`, `task_snapshots`, and `item_snapshots`, together with indexes for event time/type and robot/task history.

`attach_to_state_log()` connects warehouse event logging to persistence while preserving the state object's event list. Events and full state snapshots are recorded as the simulation runs. The dashboard retrieves recent SQLite events for its live event stream and uses the stored run identifier and snapshots for historical state information.

## 9. Live Dashboard

The dashboard is built with Python Tkinter and `ttk`, not a web framework. It provides:

- a warehouse grid map;
- true and HMM-estimated robot positions;
- robot battery, wear, and task status;
- task and inventory state;
- route and delivery visualization;
- a live SQLite-backed event stream;
- an explicit AI/ReAct trace;
- a natural-language command entry field;
- a **Run Command** button and Return-key submission;
- optional detailed debug logging;
- status and run indicators.

The simulation runs in a background thread while `_poll()` refreshes the Tkinter display approximately every 150 milliseconds. The GUI does not provide a separate interactive relaxation dialog for `ask_user`; in GUI operation, the command path falls back to the least disruptive safe relaxation when standard input is unavailable.

## 10. Multi-Robot Task Execution

The default dashboard creates two robots, two inventory items, and two warehouse tasks: `ITEM-001`/`TASK-001` and `ITEM-002`/`TASK-002`. The human command determines which items are dispatched. Verified assignments become per-robot execution queues, and robots not assigned work remain idle.

Multiple requested items can be assigned to one robot through capstone-side queue state. This is necessary because the base `Robot` model stores one `current_task`; the queue provides the multi-task execution behavior without changing that model. Tasks are assigned only after the final SMT check succeeds.

## 11. Repository Structure

### Capstone integration and supporting warehouse files

| File | Purpose |
|---|---|
| `capstone_dashboard_multi_robot.py` | Primary Tkinter dashboard and simulation lifecycle |
| `capstone_dispatch.py` | Human-request scope checks, verified dispatch, and task assignment |
| `capstone_react_agent.py` | Capstone ReAct/OpenAI integration and explicit trace |
| `capstone_smt_guard.py` | Pydantic payloads and Z3 dispatch verification |
| `capstone_memory.py` | SQLite schema, events, and snapshots |
| `capstone_main.py` | CLI-style capstone demonstration |
| `capstone_hmm_diagnostic.py` | HMM movement diagnostic |
| `orchestrator.py`, `warehouse_state.py` | Warehouse coordination and live state |
| `environment.py`, `robot.py`, `task.py`, `models.py` | Existing warehouse environment and domain models |
| `planner.py` | `ForkliftPlanner` and A* route planning |
| `conflict_resolver.py` | Minimax/alpha-beta local conflict resolution |
| `warehouse_sarsa.py`, `warehouse_rl_env.py` | SARSA training and warehouse RL environment |
| `perception.py`, `hmm_environment.py`, `hmm_filter.py` | HMM perception and filtering |


### Tests and configuration

| File | Purpose |
|---|---|
| `test_capstone_memory.py` | SQLite memory behavior |
| `test_capstone_neurosymbolic.py` | SAT, capacity, battery, and battery-drain SMT cases |
| `test_conflict_resolver.py` | Conflict resolver behavior |
| `requirements.txt` | Python package versions |
| `.env.example` | Empty `OPENAI_API_KEY` configuration placeholder |
| `.gitignore` | Ignores `.env`, logs, and `__pycache__` |

## 12. Installation

Use Python and install the repository dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

The listed packages are `z3-solver==4.13.3.0`, `pydantic==2.9.2`, `openai==1.51.2`, and `python-dotenv==1.0.1`. Tkinter, SQLite, and threading are provided outside `requirements.txt`. On Debian/Ubuntu Linux, install Tkinter for the selected Python interpreter with the system package commonly named `python3-tk` if it is absent.

## 13. Configuration

Copy `.env.example` to `.env` and set the variable `OPENAI_API_KEY` using a credential supplied through the user's own environment configuration. Never commit `.env` or expose its value. `.gitignore` is configured to ignore `.env`.

## 14. Running the Project

Start the integrated dashboard from the repository root:

```bash
python capstone_dashboard_multi_robot.py
```

The application requires Tkinter and a usable OpenAI API key with network access.

## 15. Example Commands

These examples use the item and robot identifiers created by the dashboard and the explicit assignment syntax recognized by `capstone_dispatch.py`:

```text
Assign ITEM-001 to Robot 1 and deliver it to TASK-001's destination. Use the safest feasible plan.
```

```text
Assign ITEM-001 to Robot 1 and ITEM-002 to Robot 2, then deliver both items to their task destinations.
```

```text
Assign ITEM-001 and ITEM-002 to Robot 1 and deliver both items to their task destinations.
```

```text
Assign ITEM-001 to Robot 1, but use a plan that violates the robot's capacity or battery safety constraints.
```

The final example is intended to produce a rejected or reconsidered plan when the resulting assignment is unsatisfiable. Exact natural-language interpretation remains dependent on the configured LLM; explicit item/robot mappings are checked deterministically after interpretation.

## 16. Verification and Safety Behavior

`verify_dispatch_smt()` applies the constraints implemented in `capstone_smt_guard.py`:

- each package receives exactly one robot;
- robot assignment is constrained through valid integer robot indexes;
- robots below the battery threshold receive no packages;
- package weight cannot exceed robot capacity;
- `battery_pct - drain_rate * assigned_weight >= min_battery`.

For a satisfiable model, the verifier returns `{"status": "SATISFIABLE", "assignments": ...}`. The assignment is checked against known warehouse entities, reconstructed from current state, and verified again immediately before task assignment. Only then does `WarehouseOrchestrator.assign_task()` update the live warehouse.

For an unsatisfiable model, the verifier returns `UNSATISFIABLE` diagnostics, including active robots, low-battery robots, oversized packages, total requested weight, and largest active capacity. The ReAct loop feeds verification failure back into the explicit observation path so it can reconsider, ask for a safe relaxation, or finish without authorization. Repeated verifier payloads are rejected rather than repeatedly submitted to Z3, and malformed structured output becomes a `SYNTACTIC_ERROR`. Invalid or rejected plans do not reach movement execution.

## 17. Testing

The repository contains three focused test files:

- `test_capstone_memory.py` covers SQLite memory and snapshot behavior.
- `test_capstone_neurosymbolic.py` covers satisfiable assignment, capacity rejection, low-battery rejection, and battery-drain selection.
- `test_conflict_resolver.py` covers the conflict-resolution implementation.

The SMT test file can also be run directly with `python test_capstone_neurosymbolic.py`. The repository does not contain an automated full GUI or API-backed end-to-end test. Test execution requires the Python dependencies, while API-free unit tests do not require an OpenAI request.

## 18. Known Limitations

This is a warehouse simulation rather than a physical-robot system. The sensor and movement models are simplified and stochastic. Physical movement deducts a fixed battery amount per movement, while the SMT payload supports a configurable weight-based drain expression. SARSA is retrained for each immediate waypoint, which can make execution slow. The dashboard uses hardcoded default task and reserved-goal identifiers. Task deadlines and priorities are stored/displayed but are not used by the demonstrated SMT, A*, SARSA, or scheduling decisions.

The current model stores `Task.assigned_robot`, while several capstone serialization and rendering paths read `assigned_robot_id`; consequently, some context, SQLite task snapshots, or GUI task displays may show no assigned robot even though `Task.assign()` performed the assignment. This documentation records the observed behavior and does not modify the implementation.

