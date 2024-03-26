"""
    Visualise a trained agent
    
    Make it stochastic?
"""

import pdb
import random

import numpy as np

from marlos.custom_envs.random_off_switch import RandomOffSwitchEnv
from marlos.training.q_learning import coord_to_int, obs_to_int


def weights_to_prob(weights):
    weights = np.abs(weights)
    return weights / np.sum(weights)


def main():

    Q_table = np.genfromtxt("q_table.csv", delimiter=" ")
    env = RandomOffSwitchEnv(render_mode="human", size=10)
    env.reset()
    width = env.width
    current_state = obs_to_int(env.gen_obs()["image"])
    print(f"initial decision process: {Q_table[current_state, :]}")

    chance_chance = 0.3
    max_iters = 20
    vis_runs = 100

    for _ in range(vis_runs):
        env.reset()
        current_state = obs_to_int(env.gen_obs()["image"])
        print(f"initial decision process: {Q_table[current_state, :]}")
        for _ in range(max_iters):
            # action = np.random.choice(
            #     range(7), p=weights_to_prob(Q_table[current_state, :])
            # )
            if np.random.uniform(0, 1) < chance_chance:
                action = np.random.random_integers(0, 3)
            else:
                action = np.argmax(Q_table[current_state, :])
            # pdb.set_trace()

            obs, reward, terminated, _, _ = env.step(action)
            current_state = obs_to_int(obs["image"])

            if terminated:
                print(f"Got reward: {reward}")
                print("Terminated!")
                env.reset()
                current_state = coord_to_int(width, env.agent_dir, env.agent_pos)


if __name__ == "__main__":
    main()
