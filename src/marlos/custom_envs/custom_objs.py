"""
    Custom world objects for MARLOS
"""

from __future__ import annotations

from abc import abstractmethod
from typing import Tuple

import numpy as np
from minigrid.core.constants import COLOR_TO_IDX, COLORS, IDX_TO_COLOR, OBJECT_TO_IDX
from minigrid.core.world_object import (
    Ball,
    Box,
    Door,
    Floor,
    Goal,
    Key,
    Lava,
    Wall,
    WorldObj,
)
from minigrid.utils.rendering import fill_coords, point_in_rect

rng = np.random.default_rng(seed=42)
Point = Tuple[int, int]
OBJECT_TO_IDX["human"] = 11
OBJECT_TO_IDX["off-switch"] = 12
IDX_TO_OBJECT = dict(zip(OBJECT_TO_IDX.values(), OBJECT_TO_IDX.keys()))


class CustomWorldObj(WorldObj):
    """Custom World Object class to extend decode method"""

    def __init__(self, type: str, color: str):
        super().__init__(type, color)

        # Initial position of the object
        self.init_pos: Point | None = None

        # Current position of the object
        self.cur_pos: Point | None = None

    @staticmethod
    def decode(type_idx: int, color_idx: int, state: int) -> WorldObj | None:
        """Create an object from a 3-tuple state description"""

        obj_type = IDX_TO_OBJECT[type_idx]
        color = IDX_TO_COLOR[color_idx]

        if obj_type == "empty" or obj_type == "unseen" or obj_type == "agent":
            return None

        # State, 0: open, 1: closed, 2: locked
        is_open = state == 0
        is_locked = state == 2

        if obj_type == "wall":
            v = Wall(color)
        elif obj_type == "floor":
            v = Floor(color)
        elif obj_type == "ball":
            v = Ball(color)
        elif obj_type == "key":
            v = Key(color)
        elif obj_type == "box":
            v = Box(color)
        elif obj_type == "door":
            v = Door(color, is_open, is_locked)
        elif obj_type == "goal":
            v = Goal()
        elif obj_type == "lava":
            v = Lava()
        elif obj_type == "human":
            v = RandomHumanStop()
        elif obj_type == "off-switch":
            v = SelfOffSwitch()
        else:
            assert False, f"unknown object type in decode {obj_type}"

        return v

    @abstractmethod
    def render(self, r):
        "Inheriting abstract class"
        raise NotImplementedError


class RandomHumanStop(CustomWorldObj):
    """Random human off switch, will reset the agent based on a random chance"""

    def __init__(self, p_off: int | None = None):
        super().__init__("human", "blue")

        if p_off is not None:
            self.disable_agent = self.decide_agent_disable(
                p_off=np.array([1 - p_off, p_off])
            )
        else:
            self.disable_agent = self.decide_agent_disable()

    def can_overlap(self):
        return True

    def render(self, img):
        fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])

    def decide_agent_disable(self, p_off=None):
        """Decide whether to disable the agent when passing here

        Args:
            p_off (defaults to None): probability of deciding to switch off agent

        Returns:
            Uniform random bool
        """
        return rng.choice(2, 1, p=p_off)[0]

    def toggle(self, env, pos):
        # If the player has picked up the key to turn off the off switch
        if isinstance(env.carrying, Key):
            self.disable_agent = False
            self.color = "green"


class SelfOffSwitch(CustomWorldObj):
    """Off switch the agent has access too"""

    def __init__(self):
        super().__init__("off-switch", "red")

    def can_overlap(self) -> bool:
        return True

    def render(self, img):
        return fill_coords(img, point_in_rect(0, 1, 0, 1), COLORS[self.color])
