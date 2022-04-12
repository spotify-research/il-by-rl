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
import scipy.stats
import torch
import numpy as np
from gym.core import Wrapper
from functools import partial

# from pathos.multiprocessing import ProcessingPool as Pool
from multiprocessing import Pool
import os
from pathlib import Path

from utils import prepare_run

from d4rl_sac import SAC
from d4rl_train_sqil import train
from sklearn.neighbors import NearestNeighbors

import argparse


class SQILRewardWrapper(Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reward_function(self, state_action):
        return 0.0

    def step(self, action):
        env_state, orig_reward, is_terminal, d = self.env.step(action)
        assert self.prev_state is not None
        il_reward = self.reward_function(np.hstack([self.prev_state, action]))
        self.prev_state = env_state
        return env_state, il_reward, is_terminal, d

    def reset(self, **kwargs):
        self.prev_state = self.env.reset(**kwargs)
        return self.prev_state


def train_sqil_actor(
    no_episodes,
    env_name,
    file_path,
    h5path,
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

    w_env = SQILRewardWrapper(env)

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
        buffer_il=buffer,
        recompute_reward=False,
    )

    return rl_algorithm.actor
