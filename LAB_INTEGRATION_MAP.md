# Lab -> Capstone Integration Map

| Existing work | Capstone role | Current reference implementation | Integration status |
|---|---|---|---|
| ReAct Lab | Human language -> structured action | `agent.py`, `llm_client.py`, `react_loop_lab.py`; adapted in `capstone_react_agent.py` | Integrated |
| Updated Lab | Formal safety gate + reflection | `verifier.py`; adapted in `capstone_smt_guard.py` | Integrated |
| MDP Lab | Stochastic global route planning | `warehouse_mdp.py` (new tabular value iteration over warehouse cells) | Integrated as opt-in planner; A* fallback retained. Original MDP lab source files are not in this ZIP, so this is a capstone-side implementation of the algorithm rather than a direct code port |
| Q-learning Lab | Off-policy adaptive control | `qlearning_lab.py` is a gripper/wear task; `warehouse_qlearning.py` adapts Q-learning to the navigation state used by the capstone | Integrated as optional comparison controller; original grasp lab kept separate |
| RL Prediction Lab | TD/MC value estimation | `td0_lab.py`, `mc_lab.py`, `nstep_td_lab.py` | Future diagnostics |
| Bandits Lab | Online route-strategy exploration | `route_bandit.py` (UCB1 over A* and value-iteration choices) | Integrated as opt-in route selector; not a safety authority. Original Bandits lab source files are not in this ZIP, so this is a capstone-side implementation |
| Reference warehouse | Integration architecture | `capstone_*`, `warehouse_*`, `orchestrator.py` | Base |


## New integration controls

- `WAREHOUSE_PLANNER=astar` preserves the existing default. Set it to `value_iteration` to use the stochastic grid MDP, or `bandit` to let UCB1 explore between A* and value iteration based on completed route length.
- `WarehouseOrchestrator.train_qlearning_for_target()` trains the off-policy navigation learner. `execute_qlearning_move()` only accepts the immediate next cell on the robot's current route; Q-learning cannot bypass the global route.
- The separate `qlearning_lab.py` remains a grasp/wear task and is not reused as a navigation controller because its state/action space is different.
