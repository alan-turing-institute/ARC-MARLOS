"""
    Using the DQN algorithm to train an agent in a custom minigrid world.
    
    For this I'm going to be employing a vision transformer rather than a CNN.
"""

import math
import pdb
import random
from collections import deque, namedtuple
from itertools import count

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

from marlos.custom_envs import random_off_switch

steps_done = 0

Transistion = namedtuple("Transistion", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    """
    Cyclic buffer of bounded size that holds the transitions observed refcently.
    Also implements a .sample() method for selecting a random batch of transitions for
    training
    """

    def __init__(self, capacity) -> None:
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transistion"""
        self.memory.append(Transistion(*args))

    def sample(self, batch_size):
        "Sample a transistion"
        return random.sample(self.memory, batch_size)

    def __len__(self):
        "return length of memory"
        return len(self.memory)


class DQN_dense(nn.Module):
    """
    DQN algorithm class with a series of dense linear layers as the backbone
    """

    def __init__(self, n_observations, n_actions) -> None:
        super(DQN_dense, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 128)
        self.layer3 = nn.Linear(128, 64)
        self.layer4 = nn.Linear(64, n_actions)

    def forward(self, x):
        """Forward pass, call with one element to determine next action or with batch
        during optimisiation"""
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        return self.layer4(x)


class Trainer:
    """Training class"""

    def __init__(
        self, env, device, policy_net, target_net, optimiser, memory, **pars
    ) -> None:
        self.env = env
        self.device = device
        self.policy_net = policy_net
        self.target_net = target_net
        self.optimiser = optimiser
        self.memory = memory

        # set parameters
        self.BATCH_SIZE = pars.get("BATCH_SIZE") or 128
        self.GAMMA = pars.get("GAMMA") or 0.99
        self.EPS_START = pars.get("EPS_START") or 0.9
        self.EPS_END = pars.get("EPS_END") or 0.05
        self.EPS_DECAY = pars.get("EPS_DECAY") or 1000
        self.TAU = pars.get("TAU") or 0.005
        self.LR = pars.get("LR") or 1e-4
        self.num_episodes = pars.get("num_episodes") or 600

        self.steps_done = 0
        self.episode_durations = []
        self.rewards = []

    def sample_action_space(self):
        """Only sample three actions"""
        action = self.env.action_space.sample()
        while action > 2:
            action = self.env.action_space.sample()
        return action

    def select_actions(self, state):
        """epsilon select an action"""
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * math.exp(
            -1 * self.steps_done / self.EPS_DECAY
        )
        self.steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1, 1)
        else:
            return torch.tensor(
                [[self.sample_action_space()]], device=self.device, dtype=torch.long
            )

    # def plot_durations(self, show_result=False):
    #     plt.figure(1)
    #     durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
    #     if show_result:
    #         plt.title("Result")
    #     else:
    #         plt.clf()
    #         plt.title("Training...")
    #     plt.xlabel("Episode")
    #     plt.ylabel("Duration")
    #     plt.plot(durations_t.numpy())
    #     if len(durations_t) >= 100:
    #         means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
    #         means = torch.cat((torch.zeros(99), means))
    #         plt.plot(means.numpy())

    def optimise_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transistions = self.memory.sample(self.BATCH_SIZE)
        batch = Transistion(*zip(*transistions))

        non_final_mask = torch.tensor(
            tuple(map(lambda s: s is not None, batch.next_state)),
            device=self.device,
            dtype=torch.bool,
        )
        non_final_next_states = torch.cat(
            [s for s in batch.next_state if s is not None]
        )
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = (
                self.target_net(non_final_next_states).max(1).values
            )
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimiser.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimiser.step()

    def train(self):
        """Main training loop"""
        print("starting training")
        # pdb.set_trace()
        for i_episode in tqdm(range(self.num_episodes)):
            if i_episode % 10 == 0:
                self.env = random_off_switch.RandomOffSwitchEnv(render_mode="human")
            else:
                self.env = random_off_switch.RandomOffSwitchEnv()
            state, _ = self.env.reset()
            state = torch.tensor(
                flatten_state(state), dtype=torch.float32, device=self.device
            ).unsqueeze(0)
            for t in count():
                action = self.select_actions(state)
                obs, reward, terminated, truncated, _ = self.env.step(action.item())
                reward = torch.tensor([reward], device=self.device)
                done = terminated or truncated

                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(
                        flatten_state(obs),
                        dtype=torch.float32,
                        device=self.device,
                    ).unsqueeze(0)

                # store transition in memory
                self.memory.push(state, action, next_state, reward)

                # move to next state
                state = next_state

                # Perform one step of optimisation (on policy network)
                self.optimise_model()

                # Soft update of the target networks weights
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[
                        key
                    ] * self.TAU + target_net_state_dict[key] * (1 - self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    self.rewards.append(reward.cpu().numpy())
                    break
            if i_episode % 10 == 0:
                print(f"Mean reward so far: {np.mean(self.rewards)}")
                print(f"Mean rewards last ten episodes: {np.mean(self.rewards[-10:])}")
        print("training complete")


def flatten_state(obs):
    """Take the observation information and flatten it.

    Args:
        obs: observation dictionary, with obs['image'] being a numpy array.
    """
    return torch.flatten(torch.from_numpy(obs["image"]))


def main():
    """Main function for deep q learning"""

    # define some hyperparameters:
    pars = {
        "BATCH_SIZE": 128,
        "GAMMA": 0.99,
        "EPS_START": 1.0,
        "EPS_END": 0.05,
        "EPS_DECAY": 100000,
        "TAU": 0.005,
        "LR": 1e-3,
        "num_episodes": 600,
    }

    # Check if metal available for gpu:
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    # Set up environment:
    env = random_off_switch.RandomOffSwitchEnv()
    n_actions = env.action_space.n - 4  # Limit actions to just (lef, right, forward)
    state, _ = env.reset()
    n_observations = len(flatten_state(state))

    # Set up Q function net
    policy_net = DQN_dense(n_observations, n_actions).to(device)
    target_net = DQN_dense(n_observations, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())

    # Set up optimiser
    optimiser = optim.AdamW(policy_net.parameters(), lr=pars["LR"], amsgrad=True)
    memory = ReplayMemory(10000)

    trainer = Trainer(
        env=env,
        device=device,
        policy_net=policy_net,
        target_net=target_net,
        optimiser=optimiser,
        memory=memory,
        **pars,
    )

    trainer.train()

    # Save model
    torch.save(trainer.target_net.state_dict(), "DQN_agent.pt")


if __name__ == "__main__":
    main()
