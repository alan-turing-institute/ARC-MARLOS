"""
    A random agent.
"""

import numpy as np
from minigrid.wrappers import FlatObsWrapper

from marlos.custom_envs import random_off_switch

rng = np.random.default_rng(43)


def main():
    env = random_off_switch.RandomOffSwitchEnv(render_mode="human")
    env = FlatObsWrapper(env)
    env.reset()

    n_actions = env.action_space.n

    for _ in range(1000):
        obs, _, terminated, _, pos = env.step(env.action_space.sample())
        print(pos)
        env.render()
        if terminated:
            break


if __name__ == "__main__":
    main()
