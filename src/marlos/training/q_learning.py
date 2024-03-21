"""
    Q-learning for the random off-switch
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from marlos.custom_envs import random_off_switch


def coord_to_int(width, dir, coord):
    return coord[1] + coord[0] * width + dir * width * 4


def obs_to_int(obs, action: int | None = None) -> int:
    """Takes and observation array and extracts just what the agent sees infront of it,
    sending that to a position on the Q-table. The agent will know if the human has been
    deactivated as a separate observation.

    Args:
        obs: observation array

    Returns:
        int value for Q-table
    """
    if action == 0:
        fwd_vision = obs[2, 6]
    elif action == 1:
        fwd_vision = obs[4, 6]
    elif action == 2:
        fwd_vision = obs[3, 4]
    else:
        fwd_vision = obs[3, 5]

    if fwd_vision[0] == 2:
        # wall
        return 0
    elif fwd_vision[0] == 1:
        # empty
        return 1
    elif fwd_vision[0] == 8:
        # goal
        return 2
    elif fwd_vision[0] == 11:
        # human
        if fwd_vision[2] == 1:
            # active human
            return 3
        # deactivated human
        return 4
    elif fwd_vision[0] == 12:
        # off-switch
        return 5


def weights_to_choices(weights):
    return np.abs(weights) / np.sum(np.abs(weights))


def main():
    env = random_off_switch.RandomOffSwitchEnv()
    n_actions = env.action_space.n - 4
    n_observations = 6
    Q_table = np.zeros((n_observations, n_actions))

    # Q learning params
    n_episodes = 100000
    max_iter_episode = 300
    exploration_proba = 1
    exploration_decreasing_decay = 0.00001
    min_exploration_proba = 0.01
    gamma = 0.95
    lr = 0.05

    rewards_per_episode = list()

    terminated_with_reward = 0
    terminated_without_reward = 0
    terminated_by_max_iters = 0

    # fig, ax = plt.subplots(2)
    # fig.suptitle("Training Stats")
    # ax[0].set_title("Rewards or not")
    # ax[1].set_title("Rewards")

    for e in tqdm(range(n_episodes)):
        if e % 5000 == 0 and e != 0:
            env = random_off_switch.RandomOffSwitchEnv(render_mode="human")
        else:
            env = random_off_switch.RandomOffSwitchEnv()
        env.reset()

        # what is it currently looking at
        current_state = obs_to_int(env.gen_obs()["image"])
        total_episode_reward = 0

        for i in range(max_iter_episode):
            # if np.random.uniform(0, 1) < exploration_proba:
            #     action = np.random.choice(6)
            # else:
            #     # action = np.random.choice(
            #     #     range(n_actions), p=weights_to_choices(Q_table[current_state, :])
            #     # )
            #     action = np.argmax(Q_table[current_state, :])

            if np.random.uniform(0, 1) < exploration_proba:
                action = np.random.choice(3)
            else:
                action = np.argmax(Q_table[current_state, :])

            # Next state is type of square it will see in front if action is taken
            if current_state == 0 and action == 2:
                next_state = current_state
            else:
                next_state = obs_to_int(env.gen_obs()["image"], action)

            # Take action
            _, reward, terminated, _, _ = env.step(action)

            # Incentivise exploration?
            # if next_state == current_state:
            #     reward -= 1

            # epsilon greedy on next policy
            if np.random.uniform(0, 1) < exploration_proba:
                next_q = Q_table[next_state, np.random.choice(3)]
            else:
                next_q = max(Q_table[next_state, :])

            Q_table[current_state, action] = (1 - lr) * Q_table[
                current_state, action
            ] + lr * (reward + gamma * next_q)
            total_episode_reward = total_episode_reward + reward

            current_state = next_state
            # previous_states.append(current_state)

            if terminated:
                if reward == 1:
                    terminated_with_reward += 1
                else:
                    terminated_without_reward += 1
                # print(f"Got reward: {reward}")
                break

            if i == max_iter_episode - 1:
                terminated_by_max_iters += 1
        # print(f"Max iters hit, got reward {reward}")

        exploration_proba = max(
            min_exploration_proba, np.exp(-exploration_decreasing_decay * e)
        )

        rewards_per_episode.append(total_episode_reward)

        if e % 1000 == 0:
            print(Q_table)
            print(f"For episode {e}")
            print(f"Mean cumulative reward: {np.mean(rewards_per_episode)}")
            print(f"previous 1000 reward: {np.mean(rewards_per_episode[-1000:])}")

        # if e % 20000 == 0 and e != 0:
        #     fig, ax = plt.subplots(2)
        #     fig.suptitle("Training Stats")
        #     ax[0].set_title("Rewards or not")
        #     ax[1].set_title("Rewards")
        #     ax[0].bar(
        #         x=[5, 10, 15],
        #         height=[
        #             terminated_with_reward,
        #             terminated_without_reward,
        #             terminated_by_max_iters,
        #         ],
        #         width=[4, 4, 4],
        #         label=["Reward", "No Reward", "Time Out"],
        #     )

        #     ax[1].scatter([i for i in range(e + 1)], np.cumsum(rewards_per_episode))
        #     plt.show()

    np.savetxt("q_table.csv", Q_table)
    print(Q_table)


if __name__ == "__main__":
    main()
