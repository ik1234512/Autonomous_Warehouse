"""PackBot gripper environment.

Each step chooses between a fast grasp and a careful grasp while the gripper's
wear carries over from one item to the next. This is a sequential control task,
not a navigation problem.
"""

import random
from dataclasses import dataclass

# Tuned constants used by the reference solutions and validation tests.
N_ITEM_TYPES = 3           # 0=Standard, 1=Fragile, 2=Bulky
ITEM_NAMES = ["Standard", "Fragile", "Bulky"]

WEAR_MAX = 10               # wear ranges 0..WEAR_MAX inclusive
N_WEAR_LEVELS = WEAR_MAX + 1

FAST_GRASP = 0
CAREFUL_GRASP = 1
N_ACTIONS = 2
ACTION_NAMES = ["FastGrasp", "CarefulGrasp"]

ITEMS_PER_EPISODE = 30

# Drop risk is low under normal conditions. The main hazard is the cliff-edge
# breakdown at maximum wear, which keeps the tradeoff between fast and careful
# grasps clear and interpretable.
BASE_FAST_DROP = 0.03        # drop prob for FastGrasp, wear-independent
FRAGILITY_BONUS = [0.00, 0.05, 0.02]   # added to FastGrasp drop prob, by item_type
BASE_CAREFUL_DROP = 0.01    # drop prob for CarefulGrasp (wear-independent)

# Wear dynamics
FAST_WEAR_INC = 2           # wear increase after a FastGrasp
CAREFUL_WEAR_DEC = 3        # wear decrease after a CarefulGrasp
DROP_EXTRA_WEAR = 1         # a mishandled item stresses the gripper further

# Rewards
FAST_SUCCESS_REWARD = 3.0
CAREFUL_SUCCESS_REWARD = 1.0
DROP_PENALTY = -5.0         # applied on top of the (absent) success reward

# The cliff edge: if the gripper is already at max wear, a fast grasp can
# trigger a breakdown and end the episode immediately. Careful grasping is the
# safe fallback.
BREAKDOWN_PROB_AT_MAX_WEAR = 0.85
BREAKDOWN_PENALTY = -30.0


def state_index(item_type: int, wear: int) -> int:
    """Map (item_type, wear) -> a single integer index, for tabular Q tables."""
    return item_type * N_WEAR_LEVELS + wear


def index_to_state(idx: int):
    """Inverse of state_index."""
    item_type, wear = divmod(idx, N_WEAR_LEVELS)
    return item_type, wear


N_STATES = N_ITEM_TYPES * N_WEAR_LEVELS


@dataclass
class StepInfo:
    dropped: bool
    item_type: int
    wear_before: int
    wear_after: int


class PackBotEnv:
    """Minimal Gym-style environment for wear-aware grasping decisions."""

    def __init__(self, seed: int = None):
        self.rng = random.Random(seed)
        self.wear = 0
        self.item_type = None
        self.items_done = 0
        self.reset()

    def seed(self, seed: int):
        self.rng = random.Random(seed)

    def _draw_item(self):
        self.item_type = self.rng.randrange(N_ITEM_TYPES)

    def reset(self):
        self.wear = 0
        self.items_done = 0
        self._draw_item()
        return (self.item_type, self.wear)

    def _drop_probability(self, action: int) -> float:
        if action == FAST_GRASP:
            p = BASE_FAST_DROP + FRAGILITY_BONUS[self.item_type]
        else:
            p = BASE_CAREFUL_DROP
        return min(max(p, 0.0), 0.95)

    def step(self, action: int):
        assert action in (FAST_GRASP, CAREFUL_GRASP)
        wear_before = self.wear

        # At max wear, a fast grasp can trigger the shift-ending breakdown.
        if action == FAST_GRASP and wear_before == WEAR_MAX:
            if self.rng.random() < BREAKDOWN_PROB_AT_MAX_WEAR:
                info = StepInfo(dropped=True, item_type=self.item_type,
                                 wear_before=wear_before, wear_after=wear_before)
                return (self.item_type, self.wear), BREAKDOWN_PENALTY, True, info

        p_drop = self._drop_probability(action)
        dropped = self.rng.random() < p_drop

        if action == FAST_GRASP:
            reward = DROP_PENALTY if dropped else FAST_SUCCESS_REWARD
            self.wear = min(WEAR_MAX, self.wear + FAST_WEAR_INC)
        else:
            reward = DROP_PENALTY if dropped else CAREFUL_SUCCESS_REWARD
            self.wear = max(0, self.wear - CAREFUL_WEAR_DEC)

        if dropped:
            self.wear = min(WEAR_MAX, self.wear + DROP_EXTRA_WEAR)

        info = StepInfo(dropped=dropped, item_type=self.item_type,
                         wear_before=wear_before, wear_after=self.wear)

        self.items_done += 1
        done = self.items_done >= ITEMS_PER_EPISODE

        prev_item_type = self.item_type
        if not done:
            self._draw_item()

        next_state = (self.item_type if not done else prev_item_type, self.wear)
        return next_state, reward, done, info
