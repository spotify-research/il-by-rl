# Copyright 2022 Spotify AB
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os

import numpy as np
import torch
import torch.nn as nn
from gym.core import Wrapper
from torch.optim import Adam

from d4rl_train import train
from d4rl_sac import SAC
from utils import prepare_run


class Discriminator(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(observation_size + action_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 1)

    def forward(self, x, action):
        h = torch.relu(self.fc1(torch.cat([x, action], dim=1)))
        h = torch.relu(self.fc2(h))
        # return torch.relu(self.fc3(h)) + 1e-6
        return (torch.tanh(self.fc3(h)) + 1.0) * 0.48 + 0.01


class GAILRewardWrapper(Wrapper):
    def __init__(
        self,
        env,
        expert_buffer,
        observation_size,
        action_size,
        train_iterations=500,
        learning_rate=3e-4,
        batch_size=100,
    ):
        super().__init__(env)
        self.prev_state = None
        self.train_iterations = train_iterations
        self.expert_observations, self.expert_actions = self.buffer_to_points(
            expert_buffer
        )
        self.discriminator = Discriminator(observation_size, action_size)
        self.discriminator_optim = Adam(
            self.discriminator.parameters(), lr=learning_rate
        )
        self.batch_size = batch_size

    def reward_function(self, state, action):
        disc_val = self.discriminator(
            torch.tensor(state).reshape((1, -1)), torch.tensor(action).reshape((1, -1))
        )
        reward = float(np.log(disc_val.detach()) - np.log(0.01))
        # print(disc_val.detach())
        assert np.isfinite(reward)
        return reward

    def buffer_to_points(self, buffer):
        observations = np.array(list(map(lambda x: x[0], buffer)))
        actions = np.array(list(map(lambda x: x[1], buffer)))
        return observations, actions

    def recompute_reward(self, imitator_buffer):
        imitator_observations, imitator_actions = self.buffer_to_points(imitator_buffer)
        # iterations = 500
        for _ in range(self.train_iterations):
            imitator_indices = np.random.choice(
                imitator_observations.shape[0], size=self.batch_size
            )
            expert_indices = np.random.choice(
                self.expert_observations.shape[0], size=self.batch_size
            )

            imitator_batch_obs = torch.Tensor(
                imitator_observations[imitator_indices, :]
            )
            imitator_batch_actions = torch.Tensor(imitator_actions[imitator_indices, :])

            expert_batch_obs = torch.Tensor(self.expert_observations[expert_indices, :])
            expert_batch_actions = torch.Tensor(self.expert_actions[expert_indices, :])

            loss = torch.mean(
                torch.log(
                    self.discriminator(imitator_batch_obs, imitator_batch_actions)
                )
            ) + torch.mean(
                torch.log(
                    1.0 - self.discriminator(expert_batch_obs, expert_batch_actions)
                )
            )

            self.discriminator_optim.zero_grad()
            loss.backward()
            self.discriminator_optim.step()

    def step(self, action):
        env_state, orig_reward, is_terminal, d = self.env.step(action)
        assert self.prev_state is not None
        il_reward = self.reward_function(self.prev_state, action)
        self.prev_state = env_state
        return env_state, il_reward, is_terminal, d

    def reset(self, **kwargs):
        self.prev_state = self.env.reset(**kwargs)
        return self.prev_state


def train_gail_actor(
    no_episodes,
    env_name,
    file_path,
    h5path,
    reward_recompute_interval=32,
    action_multiplier=1.0,
    gamma=0.99,
    initial_actor_path=None,
    seed=0,
    total_steps=500000,
):
    buffer, env, eval_env, smallds_expert_a_t, smallds_obs_t = prepare_run(
        env_name, file_path, h5path, no_episodes, seed
    )

    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    w_env = GAILRewardWrapper(
        env,
        buffer,
        observation_size,
        action_size,
        train_iterations=reward_recompute_interval,
    )

    rl_algorithm = SAC(observation_size, action_size, "cpu", gamma=gamma)

    if initial_actor_path is not None:
        rl_algorithm.actor.load_state_dict(torch.load(initial_actor_path))
        rl_algorithm.actor.eval()

    sac_directory = file_path
    if not os.path.exists(sac_directory):
        os.makedirs(sac_directory)
    train(
        w_env,
        eval_env,
        rl_algorithm,
        sac_directory,
        None,
        total_steps,
        buffer=buffer,
        recompute_reward=True,
        reward_recompute_interval=reward_recompute_interval,
    )

    return rl_algorithm.actor
