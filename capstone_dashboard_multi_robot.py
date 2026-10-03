"""Live dashboard for the warehouse capstone flow.

This is a thin capstone UI wrapper around the existing lab logic. It doesn't
modify the laboratory files that do the actual planning and control.

Run:
    python capstone_dashboard.py
"""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from datetime import datetime
from typing import Any


from dotenv import load_dotenv

# Load local env vars before the LLM integration imports run.
load_dotenv()

from environment import SurgicalLabEnvironment
from robot import Robot
from task import Task, InventoryItem
from warehouse_state import WarehouseState
from orchestrator import WarehouseOrchestrator

from capstone_dispatch import dispatch_verified_plan
from capstone_memory import WarehouseMemory


CELL = 34
REFRESH_MS = 150
MAX_EVENT_ROWS = 12
DEBUG_LOGGING_DEFAULT = True


class WarehouseDashboard(tk.Tk):
    def __init__(self) -> None:
        super().__init__()

        self.title("Autonomous Warehouse — Live Dashboard")
        self.geometry("1220x760")
        self.minsize(1050, 680)

        self.state: WarehouseState | None = None
        self.orchestrator: WarehouseOrchestrator | None = None
        self.memory: WarehouseMemory | None = None
        self.run_id: int | None = None
        self.worker: threading.Thread | None = None
        self.running = False
        self.completed = False
        self.debug_logging_var = tk.BooleanVar(value=DEBUG_LOGGING_DEFAULT)
        self.debug_log_path = ""
        self.debug_logger: logging.Logger | None = None
        self._batch_updating = False
        self._react_trace: list[dict[str, Any]] = []
        self.human_command_var = tk.StringVar(value="")

        self._build_ui()

        self.after(0, self._poll)

    def _build_ui(self) -> None:
        palette = {
            "ink": "#17252d",
            "rail": "#203b3f",
            "paper": "#f5f0e6",
            "card": "#fffdf8",
            "line": "#d6ccbd",
            "teal": "#2b817d",
            "coral": "#c8644f",
            "gold": "#e2b84b",
            "muted": "#718080",
        }
        self.configure(background=palette["paper"])
        self.columnconfigure(1, weight=1)
        self.rowconfigure(0, weight=1)

        rail = tk.Frame(self, bg=palette["rail"], width=190)
        rail.grid(row=0, column=0, sticky="ns")
        rail.grid_propagate(False)
        rail.rowconfigure(6, weight=1)
        tk.Label(rail, text="W/H", bg=palette["rail"], fg=palette["gold"], font=("Georgia", 20, "bold")).grid(row=0, column=0, padx=24, pady=(26, 12), sticky="w")
        tk.Label(rail, text="WAREHOUSE\nSTUDIO", bg=palette["rail"], fg="#f7f3e9", justify="left", font=("Georgia", 16, "bold")).grid(row=1, column=0, padx=24, pady=(0, 42), sticky="w")
        tk.Label(rail, text="●  LIVE OPERATIONS", bg=palette["gold"], fg=palette["ink"], anchor="w", padx=12, pady=9, font=("Consolas", 9, "bold")).grid(row=2, column=0, padx=14, sticky="ew")
        tk.Label(rail, text="○  FLEET STATUS", bg=palette["rail"], fg="#b4c9c4", anchor="w", padx=12, pady=12, font=("Consolas", 9)).grid(row=3, column=0, padx=14, sticky="ew")
        tk.Label(rail, text="○  EVENT ARCHIVE", bg=palette["rail"], fg="#b4c9c4", anchor="w", padx=12, pady=12, font=("Consolas", 9)).grid(row=4, column=0, padx=14, sticky="ew")
        tk.Label(rail, text="AUTONOMOUS\nDISPATCH\n\nA*  GLOBAL ROUTING\nRL   LOCAL CONTROL\nHMM PERCEPTION", bg=palette["rail"], fg="#7fa29d", justify="left", font=("Consolas", 8)).grid(row=7, column=0, padx=24, pady=24, sticky="sw")

        body = tk.Frame(self, bg=palette["paper"])
        body.grid(row=0, column=1, sticky="nsew", padx=28, pady=24)
        body.columnconfigure(0, weight=1)
        body.rowconfigure(2, weight=1)

        heading = tk.Frame(body, bg=palette["paper"])
        heading.grid(row=0, column=0, sticky="ew", pady=(0, 18))
        heading.columnconfigure(0, weight=1)
        tk.Label(heading, text="AUTONOMOUS WAREHOUSE", bg=palette["paper"], fg=palette["ink"], font=("Consolas", 10, "bold")).grid(row=0, column=0, sticky="w")
        tk.Label(heading, text="OPERATIONS CONSOLE  /  SHIFT 01", bg=palette["paper"], fg=palette["muted"], font=("Consolas", 9)).grid(row=1, column=0, sticky="w", pady=(7, 0))
        self.status_var = tk.StringVar(value="READY")
        self.status_label = tk.Label(heading, textvariable=self.status_var, bg=palette["paper"], fg=palette["coral"], font=("Consolas", 12, "bold"))
        self.status_label.grid(row=0, column=1, rowspan=2, sticky="e")
        self.run_var = tk.StringVar(value="Run: —")
        tk.Label(heading, textvariable=self.run_var, bg=palette["paper"], fg=palette["muted"], font=("Consolas", 8)).grid(row=2, column=1, sticky="e", pady=(8, 0))

        mission = tk.Frame(body, bg="#e6ddcd", padx=16, pady=14)
        mission.grid(row=1, column=0, sticky="ew", pady=(0, 18))
        mission.columnconfigure(1, weight=1)
        tk.Label(mission, text="MISSION INPUT", bg="#e6ddcd", fg=palette["ink"], font=("Consolas", 9, "bold")).grid(row=0, column=0, padx=(0, 14), sticky="w")
        self.command_entry = tk.Entry(mission, textvariable=self.human_command_var, bg=palette["card"], fg=palette["ink"], relief="flat", highlightthickness=1, highlightbackground=palette["line"], highlightcolor=palette["teal"], font=("Consolas", 10), insertbackground=palette["ink"])
        self.command_entry.grid(row=0, column=1, sticky="ew", padx=(0, 12), ipady=8)
        self.command_entry.bind("<Return>", lambda _event: self.start_demo())
        tk.Button(mission, text="EXECUTE", command=self.start_demo, bg=palette["coral"], fg="white", activebackground="#db806a", activeforeground="white", relief="flat", bd=0, padx=18, pady=9, font=("Consolas", 9, "bold")).grid(row=0, column=2, sticky="e")
        self.debug_path_var = tk.StringVar(value="Log: disabled until run")
        tk.Checkbutton(mission, text="ARCHIVE TRACE", variable=self.debug_logging_var, bg="#e6ddcd", fg=palette["muted"], activebackground="#e6ddcd", selectcolor=palette["card"], font=("Consolas", 8)).grid(row=1, column=0, columnspan=2, sticky="w", pady=(10, 0))
        tk.Label(mission, textvariable=self.debug_path_var, bg="#e6ddcd", fg=palette["muted"], font=("Consolas", 8)).grid(row=1, column=2, sticky="e", pady=(10, 0))

        workspace = tk.Frame(body, bg=palette["paper"])
        workspace.grid(row=2, column=0, sticky="nsew")
        workspace.columnconfigure(0, weight=4)
        workspace.columnconfigure(1, weight=2)
        workspace.rowconfigure(0, weight=1)

        map_panel = tk.Frame(workspace, bg="#e6ddcd", padx=12, pady=12)
        map_panel.grid(row=0, column=0, sticky="nsew", padx=(0, 16))
        map_panel.columnconfigure(0, weight=1)
        map_panel.rowconfigure(1, weight=1)
        tk.Label(map_panel, text="FLOORPLAN / LIVE POSITIONING", bg="#e6ddcd", fg=palette["ink"], font=("Consolas", 9, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))
        self.canvas = tk.Canvas(map_panel, background="#fffdf8", highlightthickness=1, highlightbackground=palette["line"])
        self.canvas.grid(row=1, column=0, sticky="nsew")

        telemetry = tk.Frame(workspace, bg=palette["paper"])
        telemetry.grid(row=0, column=1, sticky="nsew")
        telemetry.columnconfigure(0, weight=1)
        telemetry.rowconfigure(2, weight=1)
        telemetry.rowconfigure(3, weight=1)

        def make_card(row: int, title: str, background: str = palette["card"]):
            card = tk.Frame(telemetry, bg=background, padx=10, pady=9)
            card.grid(row=row, column=0, sticky="nsew", pady=(0, 10))
            card.columnconfigure(0, weight=1)
            tk.Label(card, text=title, bg=background, fg=palette["ink"], font=("Consolas", 9, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 6))
            return card

        robots_box = make_card(0, "FLEET TELEMETRY")
        self.robot_text = tk.Text(robots_box, height=7, width=42, state="disabled", wrap="word", bg=palette["card"], fg=palette["ink"], relief="flat", font=("Consolas", 8), padx=6, pady=4)
        self.robot_text.grid(row=1, column=0, sticky="ew")
        task_box = make_card(1, "WORK QUEUE")
        self.task_text = tk.Text(task_box, height=8, width=42, state="disabled", wrap="word", bg=palette["card"], fg=palette["ink"], relief="flat", font=("Consolas", 8), padx=6, pady=4)
        self.task_text.grid(row=1, column=0, sticky="ew")
        events_box = make_card(2, "AUDIT FEED", "#263238")
        events_box.rowconfigure(1, weight=1)
        self.event_text = tk.Text(events_box, height=10, width=42, state="disabled", wrap="none", bg="#263238", fg="#d8f3dc", relief="flat", font=("Consolas", 8), padx=6, pady=4)
        self.event_text.grid(row=1, column=0, sticky="nsew")
        react_box = make_card(3, "REASONING TRACE", "#fff8e7")
        react_box.rowconfigure(1, weight=1)
        self.react_text = tk.Text(react_box, height=10, width=42, state="disabled", wrap="word", bg="#fff8e7", fg="#674d1d", relief="flat", font=("Consolas", 8), padx=6, pady=4)
        self.react_text.grid(row=1, column=0, sticky="nsew")

    # Simulation lifecycle
    def start_demo(self) -> None:
        if self.running:
            return

        human_command = self.human_command_var.get().strip()
        if not human_command:
            messagebox.showwarning(
                "Missing human command",
                "Enter a warehouse command before running the demo.",
            )
            self.command_entry.focus_set()
            return

        # The lab client expects OPENAI_API_KEY in the environment. If it isn't
        # set, prompt once in-process and keep it local to this run.
        if not os.getenv("OPENAI_API_KEY"):
            api_key = simpledialog.askstring(
                "OpenAI API key required",
                "OPENAI_API_KEY is not set. Paste your OpenAI API key below.\n"
                "It will only be kept in this running process and is never written to the log.",
                parent=self,
                show="*",
            )
            if not api_key or not api_key.strip():
                messagebox.showwarning(
                    "OpenAI API key required",
                    "No API key was provided. The LLM dispatch cannot run without it.",
                )
                return
            os.environ["OPENAI_API_KEY"] = api_key.strip()

        self.running = True
        self.completed = False
        self.status_var.set("RUNNING")
        self.run_var.set("Run: starting...")
        self._clear_text(self.robot_text)
        self._clear_text(self.task_text)
        self._clear_text(self.event_text)
        self._clear_text(self.react_text)
        self._react_trace = []
        self.canvas.delete("all")
        self._setup_debug_logger()
        self._log_debug("Human command submitted: %r", human_command)

        # Freeze the command field while the run is active so there is only one
        # active input source.
        self.command_entry.configure(state="disabled")

        self.worker = threading.Thread(
            target=self._run_capstone,
            args=(human_command,),
            daemon=True,
        )
        self.worker.start()

    def _setup_debug_logger(self) -> None:
        """Create a per-run verbose log file when enabled in the dashboard."""
        self.debug_logger = None
        self.debug_log_path = ""
        if not self.debug_logging_var.get():
            self.debug_path_var.set("Log: disabled")
            return

        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.debug_log_path = f"capstone_debug_{stamp}.log"
        logger = logging.getLogger(f"warehouse_capstone_{id(self)}")
        logger.setLevel(logging.DEBUG)
        logger.propagate = False
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()

        handler = logging.FileHandler(self.debug_log_path, mode="w", encoding="utf-8")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(levelname)s | %(threadName)s | %(message)s"
        ))
        logger.addHandler(handler)
        self.debug_logger = logger
        self.debug_path_var.set(f"Log: {self.debug_log_path}")
        logger.info("=== CAPSTONE DEBUG LOG START ===")
        logger.info("debug_logging_enabled=%s", self.debug_logging_var.get())

    def _log_debug(self, message: str, *args: Any) -> None:
        if self.debug_logger is not None:
            self.debug_logger.info(message, *args)

    def _log_event(self, event_type: str, payload: dict[str, Any]) -> None:
        if self.debug_logger is not None:
            self.debug_logger.info("EVENT | %s | %s", event_type, payload)

    def _plan_route_avoiding_reserved_goals(
        self,
        orchestrator: WarehouseOrchestrator,
        state: WarehouseState,
        robot_id: int,
        destination: tuple[int, int],
        reserved_goals: dict[int, tuple[int, int] | set[tuple[int, int]]],
        reserved_paths: dict[int, list[tuple[int, int]]] | None = None,
    ) -> None:
        """Plan an A* route while keeping other robots' reserved goals and corridors clear."""
        env = state.environment
        blocked_cells: set[tuple[int, int]] = set()

        # Keep other robots' delivery goals out of the route planning area.
        for other_robot_id, goals in reserved_goals.items():
            if other_robot_id == robot_id:
                continue
            if isinstance(goals, tuple):
                goals = {goals}
            for goal in goals:
                if goal != destination:
                    blocked_cells.add(goal)

        # Reserve the other robot's current corridor so both robots don't try to
        # squeeze through the same path. Lower IDs keep the priority when routes overlap.
        if reserved_paths:
            other_paths = [path for rid, path in reserved_paths.items() if rid != robot_id and path]
            for path in other_paths:
                blocked_cells.update(path)

        # Never block the active robot's current cell or its destination.
        current = state.robots[robot_id].position
        blocked_cells.discard(current)
        blocked_cells.discard(destination)

        original_values: dict[tuple[int, int], int] = {}
        try:
            for x, y in blocked_cells:
                if 0 <= y < len(env.grid) and 0 <= x < len(env.grid[y]):
                    original_values[(x, y)] = env.grid[y][x]
                    env.grid[y][x] = 1

            orchestrator.plan_route(robot_id=robot_id, destination=destination)
            route = list(getattr(state.robots[robot_id], "route", None) or [])
            self._log_debug(
                "A* planned robot=%s destination=%s blocked_dynamic=%s route=%s",
                robot_id, destination, sorted(blocked_cells), route,
            )
        finally:
            for (x, y), value in original_values.items():
                env.grid[y][x] = value

    def _execute_targeted_sarsa_move(
        self,
        orchestrator: WarehouseOrchestrator,
        state: WarehouseState,
        robot_id: int,
        target: tuple[int, int],
        conflict_robot_id: int | None,
        conflict_target: tuple[int, int] | None,
        reserved_goals: dict[int, set[tuple[int, int]]],
    ) -> bool:
        """Take one SARSA step, but keep it locked to the active A* waypoint."""
        env = state.environment
        robot = state.robots[robot_id]
        self._log_debug(
            "MOVE REQUEST robot=%s pos=%s target=%s conflict_robot=%s conflict_target=%s",
            robot_id, robot.position, target, conflict_robot_id, conflict_target,
        )

        # The waypoint has to be a single step away. SARSA can suggest a local
        # action, but it cannot invent a different route from the A* plan.
        dx = target[0] - robot.position[0]
        dy = target[1] - robot.position[1]
        route_action_map = {
            (0, -1): 0,  # North
            (0, 1): 1,   # South
            (1, 0): 2,   # East
            (-1, 0): 3,  # West
        }
        route_action = route_action_map.get((dx, dy))
        if route_action is None:
            self._log_debug(
                "INVALID A* WAYPOINT robot=%s current=%s target=%s",
                robot_id, robot.position, target,
            )
            return False

        # Keep protected delivery goals out of the RL environment.
        blocked_cells: set[tuple[int, int]] = set()
        for other_id, goals in reserved_goals.items():
            if other_id == robot_id:
                continue
            blocked_cells.update(g for g in goals if g != target)

        original_values: dict[tuple[int, int], int] = {}
        try:
            for x, y in blocked_cells:
                if 0 <= y < len(env.grid) and 0 <= x < len(env.grid[y]):
                    if (x, y) == robot.position:
                        continue
                    original_values[(x, y)] = env.grid[y][x]
                    env.grid[y][x] = 1

            # Reuse the lab SARSA implementation without editing it.
            orchestrator.train_sarsa_for_target(
                robot_id=robot_id,
                target=target,
                episodes=int(os.getenv("SARSA_EPISODES", "500")),
            )
            rl_state = orchestrator.sarsa.env.reset(
                position=robot.position,
                target=target,
                battery=robot.battery,
                wear=robot.wear,
            )
            orchestrator.sarsa.epsilon = 0.0
            proposed_action = orchestrator.sarsa.choose_action(rl_state)

            action_map = {
                0: (0, -1),
                1: (0, 1),
                2: (1, 0),
                3: (-1, 0),
                4: (0, 0),
            }
            proposed_delta = action_map[proposed_action]
            proposed_position = (
                robot.position[0] + proposed_delta[0],
                robot.position[1] + proposed_delta[1],
            )

            # The policy can be checked in debug output, but it has to follow the
            # A* waypoint. We reject any move that tries to drift away from it.
            if proposed_action != route_action:
                self._log_debug(
                    "SARSA_ROUTE_GUARD robot=%s proposed_action=%s proposed_position=%s "
                    "required_action=%s required_target=%s",
                    robot_id, proposed_action, proposed_position, route_action, target,
                )
            else:
                self._log_debug(
                    "SARSA_ROUTE_MATCH robot=%s action=%s target=%s",
                    robot_id, proposed_action, target,
                )

            # Only the A*-required move is considered for collision arbitration.
            desired_position = target
            if not env.is_valid(*desired_position):
                self._log_debug(
                    "A* waypoint became blocked robot=%s target=%s",
                    robot_id, desired_position,
                )
                return False

            # Only a real collision risk triggers the local conflict resolver.
            actual_conflict_robot = None
            actual_conflict_target = None
            if conflict_robot_id is not None and conflict_robot_id != robot_id:
                other_robot = state.robots.get(conflict_robot_id)
                if other_robot is not None:
                    other_next = None
                    if conflict_target is not None:
                        other_next = conflict_target
                    risk = (
                        desired_position == other_robot.position
                        or (other_next is not None and
                            desired_position == other_next and
                            robot.position == other_robot.position)
                        or (other_next is not None and
                            other_next == robot.position and
                            desired_position == other_robot.position)
                    )
                    if risk:
                        actual_conflict_robot = conflict_robot_id
                        actual_conflict_target = conflict_target

            final_position = desired_position
            if actual_conflict_robot is not None:
                other_robot = state.robots[actual_conflict_robot]
                other_goal = actual_conflict_target
                if other_goal is None and other_robot.current_task is not None:
                    other_task = state.tasks.get(other_robot.current_task)
                    if other_task is not None:
                        other_goal = other_task.destination
                if other_goal is None:
                    other_goal = other_robot.position

                decision = orchestrator.conflict_resolver.resolve(
                    robot_position=robot.position,
                    robot_target=desired_position,
                    other_position=other_robot.position,
                    other_target=other_goal,
                )
                final_position = decision.position
                self._log_event(
                    "CONFLICT_RESOLVED",
                    {
                        "robot_id": robot_id,
                        "other_robot_id": actual_conflict_robot,
                        "proposed_action": route_action,
                        "proposed_position": desired_position,
                        "selected_action": decision.action,
                        "selected_position": final_position,
                    },
                )

            if final_position == robot.position:
                orchestrator.state.log_event(
                    "ROBOT_WAITED",
                    robot_id=robot_id,
                    reason="CONFLICT",
                )
                self._log_debug(
                    "MOVE WAIT robot=%s pos=%s reason=CONFLICT",
                    robot_id, robot.position,
                )
                return True

            if not env.is_valid(*final_position):
                self._log_debug(
                    "FINAL MOVE INVALID robot=%s final_position=%s",
                    robot_id, final_position,
                )
                return False

            orchestrator.move_robot(
                robot_id,
                final_position,
                battery_cost=1.0,
            )
            orchestrator.state.log_event(
                "SARSA_ACTION",
                robot_id=robot_id,
                action=route_action,
                proposed_action=proposed_action,
                position=state.robots[robot_id].position,
                target=target,
                controller="SARSA+ASTAR_ROUTE_GUARD",
            )
            self._log_debug(
                "MOVE RESULT robot=%s ok=True new_pos=%s battery=%.2f target=%s",
                robot_id, state.robots[robot_id].position,
                state.robots[robot_id].battery, target,
            )
            return True
        finally:
            for (x, y), value in original_values.items():
                env.grid[y][x] = value

    def _run_capstone(self, human_command: str) -> None:
        try:
            self._log_debug("Initializing custom warehouse environment")
            env = SurgicalLabEnvironment(map_type="custom")
            self._log_debug("Environment dimensions rows=%s cols=%s", len(env.grid), len(env.grid[0]))
            state = WarehouseState(env)
            orchestrator = WarehouseOrchestrator(state)
            self._log_debug("Warehouse state and orchestrator created")

            # Mirror state events into the debug log without touching the original lab logger.
            original_log_event = state.log_event
            def debug_log_event(event_type: str, **payload: Any) -> None:
                self._log_event(event_type, payload)
                original_log_event(event_type, **payload)
            state.log_event = debug_log_event  # type: ignore[method-assign]

            # Two autonomous robots.
            robot1 = Robot(
                robot_id=1,
                position=(2, 2),
                capacity=10.0,
                battery=100.0,
            )
            robot2 = Robot(
                robot_id=2,
                position=(17, 3),
                capacity=10.0,
                battery=100.0,
            )
            state.add_robot(robot1)
            state.add_robot(robot2)
            self._log_debug("Robots initialized: R1=%s R2=%s", robot1.position, robot2.position)

            memory = WarehouseMemory("warehouse_memory.db")
            run_id = memory.start_run("live-dashboard-multi-robot")
            memory.attach_to_state_log(state, run_id)
            memory.snapshot_state(state, run_id=run_id)

            self.state = state
            self.orchestrator = orchestrator
            self.memory = memory
            self.run_id = run_id
            self._log_debug("SQLite memory attached run_id=%s", run_id)

            # Create TWO items + TWO delivery tasks.
            # One job is assigned to each robot.
            #   Robot 1: ITEM-001 -> GOAL-001
            #   Robot 2: ITEM-002 -> GOAL-002
            items = [
                InventoryItem(
                    item_id="ITEM-001",
                    weight=6.0,
                    location=(2, 9),
                ),
                InventoryItem(
                    item_id="ITEM-002",
                    weight=4.0,
                    location=(15, 11),
                ),
            ]

            for item in items:
                state.add_item(item)

            tasks = [
                Task(
                    task_id="TASK-001",
                    item_id="ITEM-001",
                    pickup=(2, 9),
                    destination=(3, 4),
                    priority="URGENT",
                    deadline=20,
                ),
                Task(
                    task_id="TASK-002",
                    item_id="ITEM-002",
                    pickup=(15, 11),
                    destination=(18, 12),
                    priority="HIGH",
                    deadline=30,
                ),
            ]

            for task in tasks:
                state.add_task(task)

            self._log_debug("Loaded items=%s", [(i.item_id, i.location, i.weight) for i in items])
            self._log_debug("Loaded tasks=%s", [(t.task_id, t.item_id, t.pickup, t.destination) for t in tasks])

            # Do the dispatch only for the active run and keep the state scoped to
            # the intended robot while the check happens.
            def dispatch_one(request: str) -> dict[str, Any]:
                """Run exactly the human-requested dispatch through ReAct + SMT."""
                self._log_debug("Dispatch start request=%r", request)
                dispatch_result = dispatch_verified_plan(
                    user_request=request,
                    state=state,
                    orchestrator=orchestrator,
                )
                self._log_debug("Dispatch result: %r", dispatch_result)
                self._record_react_trace("HUMAN REQUEST", dispatch_result.get("trace", []))
                if dispatch_result["status"] != "DISPATCH_AUTHORIZED":
                    raise RuntimeError(
                        f"Dispatch failed: {dispatch_result.get('result')}"
                    )
                return dispatch_result

            # Only do the dispatch the human actually asked for.
            dispatch_result = dispatch_one(human_command)

            task_by_id = state.tasks
            item_by_id = state.inventory

            # Build the execution queues from the verified assignments only.
            robot_job_queues: dict[int, list[str]] = {1: [], 2: []}
            for assignment in dispatch_result.get("task_assignments", []):
                robot_id = int(assignment["robot_id"])
                item_id = assignment["item_id"]
                matching_tasks = [
                    task for task in task_by_id.values() if task.item_id == item_id
                ]
                if not matching_tasks:
                    raise RuntimeError(
                        f"Verified assignment references item {item_id}, but no task exists."
                    )
                robot_job_queues.setdefault(robot_id, []).append(matching_tasks[0].task_id)

            if not any(robot_job_queues.values()):
                raise RuntimeError("The verified dispatch contained no executable task assignments.")

            # Reserve each requested delivery goal so the route planner keeps the
            # other robot's target cell clear.
            reserved_goals = {
                robot_id: {task_by_id[task_id].destination for task_id in queue}
                for robot_id, queue in robot_job_queues.items()
                if queue
            }

            active_index = {1: 0, 2: 0}
            active_phase = {1: "PICKUP", 2: "PICKUP"}
            completed = {
                robot_id: not bool(robot_job_queues.get(robot_id))
                for robot_id in (1, 2)
            }

            def activate_current_task(robot_id: int) -> None:
                """Apply the active task to the live robot object as task boundaries change."""
                task_id = robot_job_queues[robot_id][active_index[robot_id]]
                state.robots[robot_id].current_task = task_id
                state.robots[robot_id].status = "MOVING"
                self._log_debug(
                    "ACTIVE_TASK robot=%s task=%s phase=%s",
                    robot_id, task_id, active_phase[robot_id],
                )

            # Keep the latest committed A* route for each robot so the planner can
            # avoid crossing the other robot's current corridor.
            committed_routes: dict[int, list[tuple[int, int]]] = {1: [], 2: []}

            def local_conflict_target(robot_id: int, waypoint: tuple[int, int] | None) -> tuple[int, int] | None:
                """Only feed the conflict resolver a real local collision risk."""
                other_id = 2 if robot_id == 1 else 1
                if completed[other_id]:
                    return None
                other_pos = state.robots[other_id].position
                other_route = committed_routes.get(other_id, [])
                other_next = other_route[1] if len(other_route) > 1 else None
                actual_risk = (
                    waypoint == other_pos
                    or (other_next is not None and waypoint == other_next and state.robots[robot_id].position == other_pos)
                    or (other_next is not None and other_next == state.robots[robot_id].position and waypoint == other_pos)
                )
                if not actual_risk:
                    return None
                other_queue = robot_job_queues.get(other_id, [])
                if not other_queue or active_index[other_id] >= len(other_queue):
                    return None
                other_task_id = other_queue[active_index[other_id]]
                return task_by_id[other_task_id].destination

            def plan_current_leg(robot_id: int) -> tuple[int, int] | None:
                """Plan the leg with A* and return only the next waypoint for SARSA."""
                task_id = robot_job_queues[robot_id][active_index[robot_id]]
                task = task_by_id[task_id]
                destination = item_by_id[task.item_id].location if active_phase[robot_id] == "PICKUP" else task.destination

                # Robot 1 gets the priority lane when both robots would otherwise
                # overlap. Robot 2 plans around that corridor.
                other_route_reservations = dict(committed_routes) if robot_id == 2 else {
                    rid: path for rid, path in committed_routes.items() if rid != robot_id
                }
                self._plan_route_avoiding_reserved_goals(
                    orchestrator=orchestrator,
                    state=state,
                    robot_id=robot_id,
                    destination=destination,
                    reserved_goals={
                        1: {task_by_id["TASK-001"].destination},
                        2: {task_by_id["TASK-002"].destination},
                    },
                    reserved_paths=other_route_reservations,
                )

                route = list(getattr(state.robots[robot_id], "route", None) or [])
                current = state.robots[robot_id].position
                if not route:
                    raise RuntimeError(
                        f"A* returned no route for robot {robot_id} from {current} to {destination}"
                    )

                # The route includes the current cell plus the next steps. If the
                # robot is already at the destination, there is nothing left to do.
                committed_routes[robot_id] = route
                waypoint = next((node for node in route[1:] if node != current), None)
                if waypoint is None:
                    self._log_debug(
                        "A* leg complete robot=%s current=%s destination=%s route=%s",
                        robot_id, current, destination, route,
                    )
                    return None

                self._log_debug(
                    "A* NEXT WAYPOINT robot=%s current=%s destination=%s waypoint=%s remaining_route=%s",
                    robot_id, current, destination, waypoint, route,
                )
                orchestrator.train_sarsa_for_target(
                    robot_id=robot_id,
                    target=waypoint,
                    episodes=int(os.getenv("SARSA_EPISODES", "500")),
                )
                return waypoint

            # Plan first pickup legs only for robots actually assigned by the command.
            next_waypoint = {1: None, 2: None}
            for robot_id in (1, 2):
                if not completed[robot_id]:
                    next_waypoint[robot_id] = plan_current_leg(robot_id)

            while not all(completed.values()):
                progress_this_round = False

                # Keep the UI in sync by batching the two robot moves into one visual tick.
                self._batch_updating = True
                try:
                    for robot_id in (1, 2):
                        if completed[robot_id]:
                            continue

                        task_id = robot_job_queues[robot_id][active_index[robot_id]]
                        activate_current_task(robot_id)
                        task = task_by_id[task_id]
                        item = item_by_id[task.item_id]
                        robot = state.robots[robot_id]
                        conflict_robot_id = 2 if robot_id == 1 else 1

                        if active_phase[robot_id] == "PICKUP":
                            if robot.position == item.location:
                                self._log_debug("PICKUP reached robot=%s task=%s item=%s at=%s", robot_id, task_id, item.item_id, robot.position)
                                orchestrator.pick_item(robot_id, item.item_id)
                                active_phase[robot_id] = "DELIVERY"
                                next_waypoint[robot_id] = plan_current_leg(robot_id)
                                progress_this_round = True
                            else:
                                next_waypoint[robot_id] = plan_current_leg(robot_id)
                                moved = self._execute_targeted_sarsa_move(
                                    orchestrator=orchestrator,
                                    state=state,
                                    robot_id=robot_id,
                                    target=next_waypoint[robot_id] or item.location,
                                    conflict_robot_id=conflict_robot_id,
                                    conflict_target=local_conflict_target(robot_id, next_waypoint[robot_id]),
                                    reserved_goals=reserved_goals,
                                )
                                if not moved:
                                    # Replan if the current route was invalidated by
                                    # the other robot or a reserved goal.
                                    plan_current_leg(robot_id)
                                    next_waypoint[robot_id] = plan_current_leg(robot_id)
                                    moved = self._execute_targeted_sarsa_move(
                                        orchestrator=orchestrator,
                                        state=state,
                                        robot_id=robot_id,
                                        target=next_waypoint[robot_id] or item.location,
                                        conflict_robot_id=conflict_robot_id,
                                        conflict_target=local_conflict_target(robot_id, next_waypoint[robot_id]),
                                        reserved_goals=reserved_goals,
                                    )
                                if not moved:
                                    raise RuntimeError(
                                        f"Robot {robot_id} could not advance toward "
                                        f"pickup {item.location}"
                                    )
                                progress_this_round = True

                        elif active_phase[robot_id] == "DELIVERY":
                            if robot.position == task.destination:
                                self._log_debug("DELIVERY reached robot=%s task=%s destination=%s", robot_id, task_id, task.destination)
                                orchestrator.deliver_item(robot_id)
                                progress_this_round = True

                                # Only move to the next task once this one is finished.
                                active_index[robot_id] += 1
                                if active_index[robot_id] >= len(robot_job_queues[robot_id]):
                                    active_phase[robot_id] = "DONE"
                                    completed[robot_id] = True
                                else:
                                    active_phase[robot_id] = "PICKUP"

                                    # The original assignment was already verified. For a
                                    # robot with multiple queued jobs, just advance the local
                                    # execution pointer instead of re-dispatching.
                                    next_waypoint[robot_id] = plan_current_leg(robot_id)
                            else:
                                next_waypoint[robot_id] = plan_current_leg(robot_id)
                                moved = self._execute_targeted_sarsa_move(
                                    orchestrator=orchestrator,
                                    state=state,
                                    robot_id=robot_id,
                                    target=next_waypoint[robot_id] or task.destination,
                                    conflict_robot_id=conflict_robot_id,
                                    conflict_target=local_conflict_target(robot_id, next_waypoint[robot_id]),
                                    reserved_goals=reserved_goals,
                                )
                                if not moved:
                                    plan_current_leg(robot_id)
                                    next_waypoint[robot_id] = plan_current_leg(robot_id)
                                    moved = self._execute_targeted_sarsa_move(
                                        orchestrator=orchestrator,
                                        state=state,
                                        robot_id=robot_id,
                                        target=next_waypoint[robot_id] or task.destination,
                                        conflict_robot_id=conflict_robot_id,
                                        conflict_target=local_conflict_target(robot_id, next_waypoint[robot_id]),
                                        reserved_goals=reserved_goals,
                                    )
                                if not moved:
                                    raise RuntimeError(
                                        f"Robot {robot_id} could not advance toward "
                                        f"delivery goal {task.destination}"
                                    )
                                progress_this_round = True

                finally:
                    self._batch_updating = False

                # Give both robots a shared visual tick before the next loop.
                time.sleep(0.18)

                if not progress_this_round:
                    raise RuntimeError("Multi-robot scheduler made no progress.")

            # Final safety check: no robot should end up on another robot's delivery goal.
            for robot_id, robot in state.robots.items():
                for other_robot_id, goal in {
                    1: task_by_id["TASK-001"].destination,
                    2: task_by_id["TASK-002"].destination,
                }.items():
                    if robot_id != other_robot_id and robot.position == goal:
                        raise RuntimeError(
                            f"Safety violation: Robot {robot_id} entered Robot "
                            f"{other_robot_id}'s delivery goal {goal}"
                        )

            # Final persistent snapshot.
            memory.snapshot_state(state, run_id=run_id)

        except Exception as exc:
            import traceback

            error_text = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))

            # Keep the traceback visible in the GUI and on disk even when the app
            # is launched without a console.
            with open("capstone_dashboard_error.log", "w", encoding="utf-8") as log:
                log.write(error_text)

            self._log_debug("UNHANDLED ERROR\n%s", error_text)
            print(error_text, flush=True)
            self.after(0, lambda e=error_text: self._show_error(e))

    def _show_error(self, error_text: str) -> None:
        self.running = False
        self.status_var.set("ERROR")
        self.command_entry.configure(state="normal")
        self._set_text(self.event_text, error_text)
        messagebox.showerror(
            "Capstone dashboard error",
            "The demo failed. Full traceback written to "
            "capstone_dashboard_error.log",
        )
    # Polling / rendering


    def _poll(self) -> None:
        try:
            if self.state is not None and not self._batch_updating:
                self._render_all()

            if (
                self.running
                and self.worker is not None
                and not self.worker.is_alive()
            ):
                self.running = False
                self.completed = True
                if self.status_var.get() != "ERROR":
                    self.status_var.set("COMPLETED")

                if self.run_id is not None:
                    self.run_var.set(f"Run: {self.run_id}")
                self._log_debug("Worker finished status=%s", self.status_var.get())
                self.command_entry.configure(state="normal")

        finally:
            self.after(REFRESH_MS, self._poll)

    def _render_all(self) -> None:
        state = self.state
        memory = self.memory
        run_id = self.run_id

        if state is None:
            return

        self._render_map(state)
        self._render_robots(state)
        self._render_tasks(state)

        if memory is not None and run_id is not None:
            self._render_events(memory, run_id)

        self._render_react_trace()

        if run_id is not None:
            self.run_var.set(f"Run: {run_id}")

    def _render_map(self, state: WarehouseState) -> None:
        env = state.environment
        rows = len(env.grid)
        cols = len(env.grid[0])

        self.canvas.config(
            width=cols * CELL + 2,
            height=rows * CELL + 2,
            scrollregion=(0, 0, cols * CELL, rows * CELL),
        )
        self.canvas.delete("all")

        # Grid + walls
        for y in range(rows):
            for x in range(cols):
                x0, y0 = x * CELL, y * CELL
                x1, y1 = x0 + CELL, y0 + CELL

                blocked = bool(env.grid[y][x])
                fill = "#4a4a4a" if blocked else "#ffffff"

                self.canvas.create_rectangle(
                    x0,
                    y0,
                    x1,
                    y1,
                    fill=fill,
                    outline="#c8c8c8",
                )

        # Item markers: only show items that are still waiting at their pickup
        # location. Once a robot picks an item up, its status becomes IN_TRANSIT
        # and the physical marker disappears from the map.
        for item in state.inventory.values():
            if getattr(item, "status", "AVAILABLE") != "AVAILABLE":
                continue

            pos = getattr(item, "location", None)
            if pos is None:
                continue

            x, y = pos
            cx = x * CELL + CELL / 2
            cy = y * CELL + CELL / 2

            self.canvas.create_oval(
                cx - 8,
                cy - 8,
                cx + 8,
                cy + 8,
                fill="#d97706",
                outline="",
                tags="item",
            )
            self.canvas.create_text(
                cx,
                cy - 12,
                text=item.item_id,
                font=("TkDefaultFont", 7, "bold"),
            )

        # Delivery goals: one green target marker per task.
        for task in state.tasks.values():
            destination = getattr(task, "destination", None)
            if destination is None:
                continue

            x, y = destination
            x0 = x * CELL + 6
            y0 = y * CELL + 6
            x1 = (x + 1) * CELL - 6
            y1 = (y + 1) * CELL - 6

            self.canvas.create_rectangle(
                x0,
                y0,
                x1,
                y1,
                outline="#16a34a",
                width=3,
            )
            self.canvas.create_text(
                x * CELL + CELL / 2,
                y * CELL + CELL / 2,
                text=f"G{str(task.task_id).split('-')[-1]}",
                fill="#166534",
                font=("TkDefaultFont", 8, "bold"),
            )

        # Robot positions + HMM estimate
        for robot in state.robots.values():
            x, y = robot.position
            cx = x * CELL + CELL / 2
            cy = y * CELL + CELL / 2

            self.canvas.create_oval(
                cx - 11,
                cy - 11,
                cx + 11,
                cy + 11,
                fill="#2563eb",
                outline="#0f172a",
                width=2,
            )
            self.canvas.create_text(
                cx,
                cy,
                text=str(robot.robot_id),
                fill="white",
                font=("TkDefaultFont", 9, "bold"),
            )

            est = getattr(robot, "estimated_position", None)
            if est is not None:
                ex, ey = est
                ecx = ex * CELL + CELL / 2
                ecy = ey * CELL + CELL / 2

                self.canvas.create_rectangle(
                    ecx - 13,
                    ecy - 13,
                    ecx + 13,
                    ecy + 13,
                    outline="#dc2626",
                    width=2,
                )

        # Route of each robot, when available.
        for robot in state.robots.values():
            route = getattr(robot, "route", None)
            if not route:
                continue

            for pos in route[1:]:
                x, y = pos
                cx = x * CELL + CELL / 2
                cy = y * CELL + CELL / 2
                self.canvas.create_oval(
                    cx - 3,
                    cy - 3,
                    cx + 3,
                    cy + 3,
                    fill="#7c3aed",
                    outline="",
                )

    def _render_robots(self, state: WarehouseState) -> None:
        lines = []

        for robot_id, robot in sorted(state.robots.items(), key=lambda t: t[0]):
            belief = getattr(robot, "belief", None)
            map_prob = float(belief.max()) if belief is not None else None

            lines.append(
                f"Robot {robot_id}\n"
                f"  true:      {robot.position}\n"
                f"  estimated: {robot.estimated_position}\n"
                f"  battery:   {robot.battery:.1f}%\n"
                f"  wear:      {robot.wear:.2f}\n"
                f"  status:    {getattr(robot, 'status', '—')}\n"
                f"  task:      {getattr(robot, 'current_task', None)}\n"
                f"  item:      {getattr(robot, 'carried_item', None)}\n"
                f"  HMM MAP:   {map_prob:.3f}"
                if map_prob is not None
                else
                f"Robot {robot_id}\n"
                f"  true:      {robot.position}\n"
                f"  estimated: {robot.estimated_position}\n"
                f"  battery:   {robot.battery:.1f}%\n"
                f"  wear:      {robot.wear:.2f}\n"
                f"  status:    {getattr(robot, 'status', '—')}"
            )

        self._set_text(self.robot_text, "\n\n".join(lines))

    def _render_tasks(self, state: WarehouseState) -> None:
        lines = []

        for task_id, task in sorted(state.tasks.items(), key=lambda t: t[0]):
            lines.append(
                f"{task_id}\n"
                f"  item:       {task.item_id}\n"
                f"  status:     {task.status}\n"
                f"  priority:   {task.priority}\n"
                f"  deadline:   {task.deadline}\n"
                f"  robot:      {getattr(task, 'assigned_robot_id', None)}"
            )

        for item_id, item in sorted(state.inventory.items(), key=lambda t: t[0]):
            lines.append(
                f"{item_id}\n"
                f"  weight:     {item.weight:.1f} kg\n"
                f"  location:   {item.location}\n"
                f"  status:     {item.status}"
            )

        self._set_text(self.task_text, "\n\n".join(lines) or "No tasks yet.")

    def _record_react_trace(
        self,
        task_id: str,
        trace: list[dict[str, str]],
    ) -> None:
        """Capture the capstone ReAct exchange for the live dashboard.

        The GUI displays the model's explicit ReAct Thought/Action summaries and
        verifier observations. Hidden model internals are never requested or shown.
        """
        for message in trace:
            role = message.get("role", "")
            content = message.get("content", "")
            if role == "assistant":
                try:
                    parsed = json.loads(content)
                except (TypeError, json.JSONDecodeError):
                    parsed = {"raw": str(content)}

                if isinstance(parsed, dict):
                    self._react_trace.append({
                        "task_id": task_id,
                        "kind": "assistant",
                        "thought": str(parsed.get("thought", "")),
                        "action": str(parsed.get("action", "")),
                        "action_input": str(parsed.get("action_input", "")),
                    })
            elif role == "user" and content:
                try:
                    observation = json.loads(content)
                except (TypeError, json.JSONDecodeError):
                    continue
                if isinstance(observation, dict) and "status" in observation:
                    self._react_trace.append({
                        "task_id": task_id,
                        "kind": "verifier",
                        "status": str(observation.get("status", "")),
                        "message": str(
                            observation.get(
                                "message",
                                observation.get("feedback", observation.get("answer", "")),
                            )
                        ),
                    })

    def _render_react_trace(self) -> None:
        if not self._react_trace:
            self._set_text(self.react_text, "Waiting for LLM / ReAct activity...")
            return

        blocks: list[str] = []
        for index, entry in enumerate(self._react_trace, start=1):
            header = f"{entry.get('task_id', 'DISPATCH')}  Step {index}"
            if entry.get("kind") == "assistant":
                thought = entry.get("thought", "").strip()
                action = entry.get("action", "").strip()
                action_input = entry.get("action_input", "").strip()
                blocks.append(
                    f"{header}\n"
                    f"Thought: {thought}\n"
                    f"Action: {action}\n"
                    f"Input: {action_input}"
                )
            else:
                blocks.append(
                    f"{header}\n"
                    f"Verifier: {entry.get('status', '')}\n"
                    f"Observation: {entry.get('message', '')}"
                )

        self._set_text(self.react_text, "\n\n".join(blocks))

    def _render_events(
        self,
        memory: WarehouseMemory,
        run_id: int,
    ) -> None:
        events = memory.recent_events(run_id, limit=MAX_EVENT_ROWS)

        lines = []
        for event in reversed(events):
            stamp = event["recorded_at"].split("T")[-1][:8]
            lines.append(
                f"{stamp}  {event['event_type']}\n"
                f"        {event['payload']}"
            )

        self._set_text(
            self.event_text,
            "\n\n".join(lines) or "Waiting for events...",
        )

    @staticmethod
    def _set_text(widget: tk.Text, text: str) -> None:
        widget.configure(state="normal")
        widget.delete("1.0", "end")
        widget.insert("1.0", text)
        widget.configure(state="disabled")

    @staticmethod
    def _clear_text(widget: tk.Text) -> None:
        WarehouseDashboard._set_text(widget, "")


def main() -> None:
    app = WarehouseDashboard()
    app.mainloop()


if __name__ == "__main__":
    main()
