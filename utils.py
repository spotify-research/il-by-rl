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

import math

import numpy as np
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import random
import copy

import gym
import d4rl_pybullet

from pathlib import Path

from itertools import chain

from d4rl_pybullet.sac import Actor
from torch.distributions import Normal
from torch.optim import Adam
from functools import partial
from multiprocessing import Pool

num_threads = 1


def get_small_dataset(percent_data, dataset_size, full_obs_t, full_expert_a_t):
    rng = np.random.RandomState(seed=1)
    data_frac = percent_data / 100.0
    small_dataset_size = math.floor(dataset_size * data_frac)
    dataset_indices = rng.choice(dataset_size, small_dataset_size, replace=False)
    smallds_obs_t = full_obs_t[dataset_indices, :]
    smallds_expert_a_t = full_expert_a_t[dataset_indices, :]
    return small_dataset_size, smallds_expert_a_t, smallds_obs_t


def execute(function):
    return function()


class Actor(nn.Module):
    def __init__(self, observation_size, action_size):
        super().__init__()
        self.fc1 = nn.Linear(observation_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.mu = nn.Linear(256, action_size)
        self.logstd = nn.Linear(256, action_size)

    def dist(self, x):
        h = torch.relu(self.fc1(x))
        h = torch.relu(self.fc2(h))
        mu = self.mu(h)
        logstd = self.logstd(h)
        clipped_logstd = logstd.clamp(-20.0, 2.0)
        return Normal(mu, clipped_logstd.exp())

    def forward(self, x, with_log_prob=False, deterministic=False, with_stdev=False):
        dist = self.dist(x)

        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()

        squashed_action, log_prob = _squash_action(dist, action)

        if with_log_prob:
            return squashed_action, log_prob

        if with_stdev:
            return squashed_action, dist.stddev

        return squashed_action


def _squash_action(dist, raw_action):
    squashed_action = torch.tanh(raw_action)
    jacob = 2 * (math.log(2) - raw_action - F.softplus(-2 * raw_action))
    log_prob = (dist.log_prob(raw_action) - jacob).sum(dim=1, keepdims=True)
    return squashed_action, log_prob


def read_model(model_path, env):
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    bc_actor = Actor(observation_size, action_size)
    bc_actor.load_state_dict(torch.load(model_path))
    bc_actor.eval()
    return bc_actor


def execute(function):
    return function()


def shuffle_episodes(dataset, seed):
    episodes = []
    episode = []
    for i in range(len(dataset["observations"])):
        obs = dataset["observations"][i, :]
        reward = float(dataset["rewards"][i])
        terminal = float(dataset["terminals"][i])
        action = dataset["actions"][i, :]
        episode.append((obs, reward, terminal, action))
        if terminal:
            episodes.append(episode)
            episode = []

    random.seed(seed)
    random.shuffle(episodes)

    dataset = {"observations": [], "rewards": [], "terminals": [], "actions": []}
    for episode in episodes:
        for (obs, reward, terminal, action) in episode:
            dataset["observations"].append(obs)
            dataset["rewards"].append(reward)
            dataset["terminals"].append(terminal)
            dataset["actions"].append(action)

    dataset["observations"] = np.array(dataset["observations"])
    dataset["actions"] = np.array(dataset["actions"])
    dataset["rewards"] = np.array(dataset["rewards"]).flatten()
    dataset["terminals"] = np.array(dataset["terminals"]).flatten()

    return dataset


def evaluate_dataset(dataset):
    episode_returns = []
    episode_return = 0
    for i in range(len(dataset["observations"])):
        reward = float(dataset["rewards"][i])
        terminal = float(dataset["terminals"][i])
        episode_return += reward
        if terminal:
            episode_returns.append(episode_return)
            episode_return = 0

    return np.array(episode_returns).mean()


def populate_buffer(dataset, no_episodes):
    buffer = []
    obs_all = []
    actions_all = []
    episode = 0
    for i in range(len(dataset["observations"])):
        obs_t = (dataset["observations"][i, :]).flatten()
        act_t = (dataset["actions"][i, :]).flatten()
        # rew_t = torch.Tensor(dataset['rewards'][i])
        rew_t = 1.0
        ter_t = bool(dataset["terminals"][i])
        buffer.append([obs_t, act_t, [rew_t], [ter_t]])
        obs_all.append(obs_t)
        actions_all.append(act_t)
        if ter_t:
            episode += 1
            if episode >= no_episodes:
                break

    if no_episodes < 1.0:  # for fractional episodes, we keep the first part
        keep_data = math.floor(no_episodes * len(buffer))
        buffer = buffer[:keep_data]
        obs_all = obs_all[:keep_data]
        actions_all = actions_all[:keep_data]

    return buffer, np.array(obs_all), np.array(actions_all)


def test_policy(bc_actor, env, episodes=100, deterministic=True):
    episode_rewards = []
    for i in range(episodes):
        # env.render(mode='human', close=False)
        env_state = env.reset()
        is_terminal = False
        total_reward = 0.0
        steps = 0
        while not is_terminal:

            env_state_t = torch.Tensor(env_state.reshape((1, len(env_state))))
            bc_action = (
                bc_actor(env_state_t, deterministic=deterministic)
                .data.numpy()
                .flatten()
            )
            env_state, reward, is_terminal, _ = env.step(bc_action)
            total_reward += reward
            steps += 1

        print(f"Total Reward: {total_reward:.4f}, Total steps: {steps}.")
        episode_rewards.append(total_reward)
    return episode_rewards


def test_policy_once(iteration, bc_actor, env, deterministic):
    np.random.seed(iteration)
    random.seed(iteration)
    env.seed(iteration)
    return test_policy(bc_actor, env, episodes=1, deterministic=deterministic)[0]


def test_policy_parallel(bc_actor, env, episodes=100, deterministic=True):
    with Pool(processes=60) as pool:
        episode_rewards = list(
            pool.map(
                partial(
                    test_policy_once,
                    bc_actor=bc_actor,
                    env=env,
                    deterministic=deterministic,
                ),
                range(episodes),
            )
        )

    return episode_rewards


def generate_tasks(callable, method_name, env_name, h5path):
    tasks = []
    for seed in [1, 2, 3, 4, 5]:
        for ep in [1, 2, 4, 8, 16]:  # [32,64,128,256,512]:
            tasks.append(
                partial(
                    callable,
                    no_episodes=ep,
                    env_name=env_name,
                    seed=seed,
                    h5path=h5path,
                    file_path=f"{env_name}/{method_name}_model_ep{ep}_seed{seed}",
                )
            )

        for ep in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
            tasks.append(
                partial(
                    callable,
                    no_episodes=1 / ep,
                    env_name=env_name,
                    seed=seed,
                    h5path=h5path,
                    file_path=f"{env_name}/{method_name}_model_ep1_{ep}_seed{seed}",
                )
            )
    return tasks


env_paths = {
    "ant": ("ant-bullet-medium-v0", "./datasets/AntBulletEnv-v0_medium_0/buffer.hdf5"),
    "hopper": (
        "hopper-bullet-medium-v0",
        "./datasets/HopperBulletEnv-v0_medium_0/buffer.hdf5",
    ),
    "walker": (
        "walker2d-bullet-medium-v0",
        "./datasets/Walker2DBulletEnv-v0_medium_0/buffer.hdf5",
    ),
    "halfcheetah": (
        "halfcheetah-bullet-medium-v0",
        "./datasets/HalfCheetahBulletEnv-v0_medium_0/buffer.hdf5",
    ),
}


def prepare_run(env_name, file_path, h5path, no_episodes, seed):
    torch.set_num_threads(num_threads)
    env = gym.make(env_name)
    eval_env = gym.make(env_name)
    Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
    dataset = env.get_dataset(h5path=h5path)
    dataset = shuffle_episodes(dataset, seed=seed)
    buffer, smallds_obs_t, smallds_expert_a_t = populate_buffer(dataset, no_episodes)
    return buffer, env, eval_env, smallds_expert_a_t, smallds_obs_t


class SimpleLogger:
    def __init__(self, logdir):
        self.logdir = logdir

    def add(self, name, step, value):
        with open(os.path.join(self.logdir, name + ".csv"), "a") as f:
            print("%d,%f" % (step, value), file=f)

        print("step=%d %s=%f" % (step, name, value))

    def add2(self, name, step, value1, value2):
        with open(os.path.join(self.logdir, name + ".csv"), "a") as f:
            print("%d,%f,%f" % (step, value1, value2), file=f)

        print("step=%d val1=%f val2=%f" % (step, value1, value2))
