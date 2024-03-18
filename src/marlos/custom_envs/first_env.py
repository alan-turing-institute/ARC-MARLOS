"""
    Creating my first custom environment for playing around in
"""

from __future__ import annotations

from abc import abstractmethod

from minigrid.core.constants import COLOR_NAMES
from minigrid.core.grid import Grid
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Key, Wall
from minigrid.manual_control import ManualControl
from minigrid.minigrid_env import MiniGridEnv


class SimpleEnv(MiniGridEnv):

    def __init__(
        self,
        size=8,
        agent_start_pos=(1, 1),
        agent_start_dir=0,
        max_steps: int | None = None,
        **kwargs,
    ):
        self.agent_start_pos = agent_start_pos
        self.agent_start_dir = agent_start_dir

        mission_space = MissionSpace(mission_func=self._gen_mission)

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            max_steps=256,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "simple mission"

    @abstractmethod
    def _gen_grid(self, width, height):

        # create the grid
        self.grid = Grid(width, height)
        self.grid.wall_rect(0, 0, width, height)

        # place a wall
        for i in range(0, height):
            self.grid.set(5, i, Wall())

        # place door
        self.grid.set(5, 6, Door(COLOR_NAMES[0], is_locked=True))
        self.grid.set(3, 6, Key(COLOR_NAMES[0]))

        # place goal
        self.put_obj(Goal(), width - 2, height - 2)

        # place agent
        if self.agent_start_pos is not None:
            self.agent_pos = self.agent_start_pos
            self.agent_dir = self.agent_start_dir
        else:
            self.place_agent()
            self.place_agent()
            self.place_agent()


def main():
    env = SimpleEnv(render_mode="human")
    # env.reset()

    # Enable manual control for testing
    manual_control = ManualControl(env, seed=42)
    manual_control.start()


if __name__ == "__main__":
    main()
