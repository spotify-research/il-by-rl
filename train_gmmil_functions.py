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

# from pathos.multiprocessing import ProcessingPool as Pool
import os

import numpy as np
import scipy.stats
import torch
from gym.core import Wrapper
from sklearn.neighbors import NearestNeighbors

from d4rl_train import train
from d4rl_sac import SAC
from utils import prepare_run


class GMMILRewardWrapper(Wrapper):
    def __init__(self, env, expert_buffer, action_multiplier=1.0):
        super().__init__(env)
        self.action_multiplier = action_multiplier
        self.prev_state = None
        self.neighbors_considered = min(8, len(expert_buffer))
        expert_points = self.buffer_to_points(expert_buffer)
        self.nbrs_expert = NearestNeighbors(
            n_neighbors=self.neighbors_considered, algorithm="ball_tree"
        ).fit(expert_points)
        self.kernel_bandwidth_expert = self.bandwidth_from_points(expert_points)
        print(f"Expert bandwidth is {self.kernel_bandwidth_expert}")
        self.kernel_bandwidth_initial = None
        self.recompute_reward(imitator_buffer=[])

    def bandwidth_from_points(self, points):
        if points.shape[0] > 1:
            distances = scipy.spatial.distance.pdist(points)
            return float(np.median(distances.flatten()))
        else:
            return 1.0

    def reward_function(self, state_action):
        distances_expert, _ = self.nbrs_expert.kneighbors(state_action.reshape((1, -1)))
        kernel_vals_expert = self.kernel_function(distances_expert)

        if self.nbrs_imitator:
            distances_imitator, _ = self.nbrs_imitator.kneighbors(
                state_action.reshape((1, -1))
            )
            kernel_vals_imitator = self.kernel_function(distances_imitator)
            reward = float(np.mean(kernel_vals_expert) - np.mean(kernel_vals_imitator))
        else:
            reward = float(np.mean(kernel_vals_expert))

        assert np.isfinite(reward)
        return reward

    def kernel_function(self, distance):
        if self.kernel_bandwidth_initial:
            return 0.5 * (
                scipy.stats.norm.pdf(
                    distance, loc=0, scale=self.kernel_bandwidth_expert
                )
                + scipy.stats.norm.pdf(
                    distance, loc=0, scale=self.kernel_bandwidth_initial
                )
            )
        else:
            return scipy.stats.norm.pdf(
                distance, loc=0, scale=self.kernel_bandwidth_expert
            )

    def buffer_to_points(self, buffer):
        observations = np.array(list(map(lambda x: x[0], buffer)))
        actions = np.array(list(map(lambda x: x[1], buffer)))
        points = np.hstack((observations, actions))
        return points

    def recompute_reward(self, imitator_buffer):
        imitator_points = self.buffer_to_points(imitator_buffer)
        if imitator_points.shape[0] > self.neighbors_considered:
            self.nbrs_imitator = NearestNeighbors(
                n_neighbors=self.neighbors_considered, algorithm="ball_tree"
            ).fit(imitator_points)
            if not self.kernel_bandwidth_initial:  # only set this once
                self.kernel_bandwidth_initial = self.bandwidth_from_points(
                    imitator_points
                )
                print(
                    f"Bandwidth computed from points initially in the buffer is {self.kernel_bandwidth_initial}"
                )
        else:
            self.nbrs_imitator = None

    def step(self, action):
        env_state, orig_reward, is_terminal, d = self.env.step(action)
        assert self.prev_state is not None
        il_reward = self.reward_function(
            np.hstack([self.prev_state, self.action_multiplier * action])
        )
        self.prev_state = env_state
        return env_state, il_reward, is_terminal, d

    def reset(self, **kwargs):
        self.prev_state = self.env.reset(**kwargs)
        return self.prev_state


def train_gmmil_actor(
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

    w_env = GMMILRewardWrapper(env, buffer, action_multiplier)

    initial_reward_scaling = float(w_env.kernel_function(0.0))
    for datapoint in buffer:
        datapoint[2][0] *= initial_reward_scaling

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
