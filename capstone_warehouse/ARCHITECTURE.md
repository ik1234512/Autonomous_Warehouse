# Smart Autonomous Warehouse — Architecture

## 1. Architecture Summary

The project is a Track A warehouse simulation. Its integrated entry point is the Tkinter dashboard in `capstone_dashboard_multi_robot.py`. The dashboard accepts a natural-language command, obtains an explicit structured ReAct result from an OpenAI-backed agent, validates that result with Pydantic and Z3, and authorizes only satisfiable assignments. The authorized tasks then pass through the existing warehouse orchestrator, A* route planner, SARSA local controller, HMM perception layer, conditional conflict resolver, and SQLite memory layer.

The layers are deliberately separated: the LLM interprets intent, Z3 makes the dispatch safety decision, A* supplies global route authority, SARSA proposes local adaptive movement, and HMM perception estimates position from noisy observations. Tkinter presents the resulting state and explicit recorded trace fields.

## 2. Component Diagram

```text
Tkinter human input
        |
        v
WarehouseReActAgent  -- OpenAI chat completion
        |
        v
Pydantic AgentStep / explicit trace
        |
        v
capstone_dispatch.py
        |
        v
capstone_smt_guard.py -- Z3
     | SAT                         | UNSAT / invalid
     v                             v
WarehouseOrchestrator        ReAct observation and
verified task queues          no execution authorization
     |
     +--> ForkliftPlanner.a_star() --> route/reservations
     +--> WarehouseSARSA ------------> local action
     +--> ConflictResolver ----------> immediate risk decision
     |
     v
Robot / SurgicalLabEnvironment
     |
     +--> RobotPerception + HMM estimate
     +--> WarehouseState events
     +--> WarehouseMemory / SQLite
     +--> Dashboard polling and rendering
```

## 3. End-to-End Data Flow

1. `WarehouseDashboard` stores human text in `human_command_var` and starts `_run_capstone()` from `start_demo()`.
2. The capstone creates `SurgicalLabEnvironment`, `WarehouseState`, `WarehouseOrchestrator`, two `Robot` objects, inventory items, tasks, and `WarehouseMemory`.
3. `dispatch_verified_plan()` builds authoritative context and calls `WarehouseReActAgent.run()`.
4. Each response is parsed as Pydantic `AgentStep`. ReAct may invoke `verify_dispatch_smt`, `ask_user`, or `finish`.
5. SAT assignments are scope-checked against the human request and known entities, then submitted to a final SMT check.
6. Only the final SAT result is converted into task assignments and per-robot queues.
7. `WarehouseOrchestrator.plan_route()` calls `ForkliftPlanner.a_star()` for the current leg.
8. SARSA proposes a local action. The dashboard execution path accepts it only when it remains compatible with the A* route and reservation/goal checks.
9. Immediate collision or swap risk invokes `ConflictResolver`.
10. Movement updates robot state, battery, time, HMM perception, and event persistence. Pickup and delivery update item and task state.
11. Tkinter `_poll()` refreshes the map, robot/task panels, event stream, and ReAct trace.

## 4. Component Responsibilities

**Human Interface.** `WarehouseDashboard` provides the Tkinter command field, Run Command button, status display, map, robot/task panels, event stream, debug-log option, and AI/ReAct trace.

**ReAct/LLM.** `WarehouseReActAgent` in `capstone_react_agent.py` calls `openai.OpenAI().chat.completions.create(...)`, using `gpt-5-mini` by default. It limits the loop to eight turns and records explicit trace data. The supported actions are `verify_dispatch_smt`, `ask_user`, and `finish`.

**Pydantic.** `AgentStep` and related models validate the structured agent response. Malformed output becomes a `SYNTACTIC_ERROR` instead of being treated as an assignment.

**SMT/Z3 verifier.** `verify_dispatch_smt()` in `capstone_smt_guard.py` checks exactly-one package assignment, valid robot indexes, battery-threshold eligibility, capacity, and post-assignment battery drain. It returns SAT assignments or UNSAT diagnostics.

**Task dispatch.** `capstone_dispatch.py` enforces explicit item/robot intent, rejects unknown entities, reconstructs a current-state payload, performs the final verification, and calls `WarehouseOrchestrator.assign_task()` only after SAT.

**HMM perception.** `RobotPerception`, `HMMSensor`, and `HMMFilter` model noisy observations and estimate hidden position. The sensor accuracy is 80%; the transition model uses intended movement at 80% and lateral slips at 10% each.

**A*.** `ForkliftPlanner.a_star()` plans four-direction grid routes with unit cost and a Manhattan heuristic while accounting for obstacles and capstone route/goal restrictions.

**Conflict Resolver.** `ConflictResolver` uses depth-2 Minimax with alpha-beta pruning. Its evaluation considers collision, swap, proximity, and waiting penalties. The integrated path invokes it for immediate collision or swap risk.

**SARSA.** `WarehouseSARSA` trains with `WarehouseRLEnv` and supports north, south, east, west, and wait actions. It is trained per immediate waypoint and proposes local movement beneath the A* route authority.

**Robot/environment.** `Robot`, `SurgicalLabEnvironment`, `WarehouseState`, and `WarehouseOrchestrator` hold simulated positions, inventory, tasks, movement, and delivery state.

**SQLite memory.** `WarehouseMemory` creates `runs`, `events`, `robot_snapshots`, `task_snapshots`, and `item_snapshots`, with history-oriented indexes. State events are captured through `attach_to_state_log()`.

**Dashboard.** Tkinter polling reads live state and recent SQLite events, then renders true/estimated positions, routes, status, tasks, events, and the explicit AI trace.

## 5. Safety and Verification Flow

```text
LLM structured step
        |
        v
Pydantic validation
        |
        +-- malformed --> SYNTACTIC_ERROR / no dispatch
        |
        v
verify_dispatch_smt()
        |
        +-- UNSAT --> diagnostics -> ReAct observation -> reconsider,
        |             ask for safe relaxation, or finish without authorization
        |
        +-- SAT --> scope/entity checks -> final SMT check
                                      |
                                      +-- fail --> abort before execution
                                      |
                                      +-- SAT --> assign tasks and execute
```

Explicit human mappings such as `ITEM-001 to Robot 1` are checked after interpretation, so the LLM cannot silently change the named robot. Repeated verifier payloads are rejected without repeatedly running Z3. The verified assignment is preserved into the execution queues.

## 6. Multi-Robot Coordination

The dashboard initializes two robots and two tasks. A human request selects the items that enter dispatch. The SAT result assigns each requested item to a robot, after which the capstone maintains per-robot queues. Multiple tasks may therefore be queued for one robot even though the base `Robot` object stores one `current_task`.

A* routes use grid obstacles and capstone protected delivery goals/reserved routes. The local execution step checks that SARSA does not leave the active A* route or enter a protected/reserved location. The conflict resolver is then used when the next movement presents an immediate same-cell collision or swap risk. This capability exists independently of whether a particular default run produces such a conflict.

## 7. Data Persistence

`WarehouseMemory("warehouse_memory.db")` records run records, event history, robot snapshots, task snapshots, and item snapshots. Events are attached to `WarehouseState.log_event()` and snapshots capture the state around those events. The dashboard reads recent events for its live event stream. The database is a runtime artifact and should be treated separately from the source modules.

## 8. GUI Data Flow

The worker thread updates the live warehouse objects and memory. Tkinter remains responsible for presentation: `_poll()` periodically reads the current state, route data, estimated positions, task/item status, recent SQLite events, and `_react_trace`. Canvas drawing presents the map and robot positions; text widgets present robot status, tasks/inventory, events, and the explicit ReAct trace. The GUI has no headless mode and requires Tkinter.

## 9. Design Rationale

Natural language is useful for expressing high-level warehouse intent but should not directly actuate a robot. Therefore, the LLM is limited to interpretation and explicit tool-loop decisions, while Z3 provides a deterministic dispatch gate. HMM filtering represents uncertainty in perception, A* provides an interpretable global route, and SARSA adapts local movement using a defined action/reward environment. Conflict resolution addresses multi-robot interactions separately from route generation. SQLite and Tkinter provide auditability and visibility across the run.

## 10. Rubric Coverage

| Essential layer | Components |
|---|---|
| Human Interface & Safety Filter | `capstone_dashboard_multi_robot.py`, `capstone_react_agent.py`, `capstone_dispatch.py`, `capstone_smt_guard.py` |
| Perception | `perception.py`, `hmm_environment.py`, `hmm_filter.py` |
| Route Finder & Conflict Resolver | `planner.py`, `orchestrator.py`, `conflict_resolver.py` |
| Adaptive Pilot | `warehouse_sarsa.py`, `warehouse_rl_env.py` |
| Memory | `capstone_memory.py`, `warehouse_state.py`, SQLite runtime database |
| Live Dashboard | `capstone_dashboard_multi_robot.py` |

The integrated system is the Track A warehouse benchmark path. The repository also contains separate legacy/laboratory exercises; their presence does not imply that every file participates in the live dashboard pipeline.
