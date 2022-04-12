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

import copy

import gym

import torch
import numpy as np
from gym.core import Wrapper

import os

from utils import prepare_run

from d4rl_sac import SAC
from d4rl_train import train
from sklearn.neighbors import NearestNeighbors


class ILRewardWrapper(Wrapper):
    def __init__(self, env, env_name, is_in_support, action_multiplier=1.0):
        super().__init__(env)
        self.is_in_support = is_in_support
        self.action_multiplier = action_multiplier
        self.env_name = env_name
        self.prev_state = None

    def step(self, action):
        env_state, orig_reward, is_terminal, d = self.env.step(action)
        assert self.prev_state is not None
        il_reward = self.is_in_support(
            np.hstack([self.prev_state, self.action_multiplier * action])
        )
        self.prev_state = env_state
        return env_state, il_reward, is_terminal, d

    def reset(self, **kwargs):
        self.prev_state = self.env.reset(**kwargs)
        return self.prev_state

    def clone(self):
        new_env = gym.make(self.env_name)
        new_env = ILRewardWrapper(
            new_env,
            self.env_name,
            copy.deepcopy(self.is_in_support),
            self.action_multiplier,
        )
        new_env.prev_state = copy.deepcopy(self.prev_state)
        return new_env


def make_buffer_next(buffer):
    buffer_next = copy.deepcopy(buffer)
    buffer_next.pop(0)
    return buffer_next


def train_il_actor(
    no_episodes,
    env_name,
    file_path,
    h5path,
    action_multiplier=1.0,
    gamma=0.99,
    initial_actor_path=None,
    eval2=False,
    eval_interval=10000,
    seed=0,
    total_steps=500000,
):
    buffer, env, eval_env, smallds_expert_a_t, smallds_obs_t = prepare_run(
        env_name, file_path, h5path, no_episodes, seed
    )

    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    is_in_support = get_binary_reward(
        smallds_expert_a_t, smallds_obs_t, action_multiplier
    )
    w_env = ILRewardWrapper(env, env_name, is_in_support, action_multiplier)

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
        eval2=eval2,
        eval_interval=eval_interval,
    )

    # torch.save(rl_algorithm.actor.state_dict(), file_path)
    return rl_algorithm.actor


def get_binary_reward(smallds_expert_a_t, smallds_obs_t, action_multiplier):
    smallds_obs_action_t = np.hstack(
        [smallds_obs_t, action_multiplier * smallds_expert_a_t]
    )
    is_in_support = support_estimator(smallds_obs_action_t)
    assert is_in_support(smallds_obs_action_t[0, :]) > 0.0
    return is_in_support


def support_estimator(points):
    tolerance = 0.0
    nbrs1 = NearestNeighbors(n_neighbors=1, algorithm="ball_tree").fit(points)

    def is_in_support(x):
        distance, _ = nbrs1.kneighbors(x.reshape((1, len(x))))
        distance = float(distance)
        if distance <= tolerance:
            return 1.0
        how_far = distance - tolerance  # > 0
        return 1.0 - how_far**2  # was 1.0 - how_far ** 2

    return is_in_support
