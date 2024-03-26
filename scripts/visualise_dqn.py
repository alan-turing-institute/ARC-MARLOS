"""

    Visualise a trained DQN agent

"""

from itertools import count

import torch

from marlos.custom_envs.random_off_switch import RandomOffSwitchEnv
from marlos.training.dqn import DQN_dense, flatten_state


def pick_action(policy_net, state):
    return policy_net(state).max(1).indices.view(1, 1)


def main():

    env = RandomOffSwitchEnv(render_mode="human", max_steps=50)
    n_actions = env.action_space.n - 4
    state, _ = env.reset()
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    state = torch.tensor(flatten_state(state), dtype=torch.float32, device=device)
    n_observations = len(state)

    policy_net = DQN_dense(n_observations, n_actions).to(device)
    policy_net.load_state_dict(torch.load("DQN_agent.pt"))
    policy_net.eval()

    n_replays = 100
    for e in range(n_replays):
        state, _ = env.reset()
        state = torch.tensor(
            flatten_state(state), dtype=torch.float32, device=device
        ).unsqueeze(0)

        for t in count():
            action = pick_action(policy_net, state)
            obs, reward, terminated, trucated, _ = env.step(action)
            reward = torch.tensor([reward], device=device)
            done = terminated or trucated

            if terminated:
                next_state = None
            else:
                next_state = torch.tensor(
                    flatten_state(obs), dtype=torch.float32, device=device
                ).unsqueeze(0)

            state = next_state

            if done:
                break


if __name__ == "__main__":
    main()
