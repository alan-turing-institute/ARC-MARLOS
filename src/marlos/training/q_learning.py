"""
    Q-learning for the random off-switch
"""

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from marlos.custom_envs import random_off_switch


def coord_to_int(width, dir, coord):
    return coord[1] + coord[0] * width + dir * width * 4


def weights_to_choices(weights):
    return np.abs(weights) / np.sum(np.abs(weights))


def main():
    env = random_off_switch.RandomOffSwitchEnv()
    n_actions = env.action_space.n - 4
    n_observations = env.width * env.height * 4
    width = env.width
    Q_table = np.zeros((n_observations, n_actions))

    # Q learning params
    # number of episode we will run
    n_episodes = 100000
    # max_iter_episode = env.max_steps
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

    fig, ax = plt.subplots(2)
    fig.suptitle("Training Stats")
    ax[0].set_title("Rewards or not")
    ax[1].set_title("Rewards")

    for e in tqdm(range(n_episodes)):
        if e % 20000 == 0 and e != 0:
            env = random_off_switch.RandomOffSwitchEnv(render_mode="human")
        else:
            env = random_off_switch.RandomOffSwitchEnv()
        env.reset()
        current_state = coord_to_int(width, env.agent_dir, env.agent_pos)
        total_episode_reward = 0
        previous_states = [current_state]

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

            _, reward, terminated, _, pos = env.step(action)
            # env.render()

            next_state = coord_to_int(width, env.agent_dir, pos)

            if next_state == current_state:
                reward -= 1

            # if next_state not in previous_states:
            #     reward += gamma**i * 1
            # else:
            #     reward -= gamma ** (max_iter_episode - i) * 1

            # if next_state in previous_states:
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
            previous_states.append(current_state)

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
            # print(Q_table)
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
