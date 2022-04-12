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
from pathlib import Path

import gym
import d4rl_pybullet
import numpy as np
import torch
from matplotlib import pyplot as plt

from utils import Actor, prepare_run, test_policy


def train_bc_actor(
    no_episodes, file_path, env_name, h5path, seed=0, total_steps=500000
):
    buffer, env, eval_env, smallds_expert_a_t, smallds_obs_t = prepare_run(
        env_name, file_path, h5path, no_episodes, seed
    )
    observation_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    batch_size_bc = 100

    small_dataset_size = smallds_obs_t.shape[0]

    smallds_obs_t = torch.Tensor(smallds_obs_t)
    smallds_expert_a_t = torch.Tensor(smallds_expert_a_t)

    bc_actor = Actor(observation_size, action_size)
    optimizer = torch.optim.Adam(bc_actor.parameters(), lr=3e-4)

    losses = []
    rng = np.random.RandomState(seed=seed)
    for _ in range(total_steps // batch_size_bc):
        current_loss = optimization_step(
            bc_actor,
            batch_size_bc,
            losses,
            optimizer,
            rng,
            small_dataset_size,
            smallds_expert_a_t,
            smallds_obs_t,
        )
        # print(f"Loss: {current_loss:.4f}")

    # plt.plot(np.array(losses))
    # plt.show()
    # torch.save(bc_actor.state_dict(), file_path)

    evaluations = test_policy(bc_actor, eval_env, episodes=10)
    average_return = float(np.array(evaluations).mean())
    print(f"Average return {average_return}")
    reward_file_path = file_path + "/eval_reward.csv"
    Path(os.path.dirname(reward_file_path)).mkdir(parents=True, exist_ok=True)
    with open(reward_file_path, "w") as file:
        file.write(f"0,{average_return}")

    return bc_actor


def optimization_step(
    bc_actor,
    batch_size_bc,
    losses,
    optimizer,
    rng,
    small_dataset_size,
    smallds_expert_a_t,
    smallds_obs_t,
):
    optimizer.zero_grad()
    bs = batch_size_bc
    minibatch_indices = rng.choice(small_dataset_size, bs, replace=True)
    obs_t = smallds_obs_t[minibatch_indices, :]
    expert_a_t = smallds_expert_a_t[minibatch_indices, :]
    actor_act_t, act_stddev_t = bc_actor(
        obs_t, deterministic=True, with_log_prob=False, with_stdev=True
    )
    target_sigma = 0.01
    per_state_losses = torch.mean((expert_a_t - actor_act_t) ** 2, axis=1) + torch.mean(
        (target_sigma - act_stddev_t) ** 2, axis=1
    )
    loss = torch.mean(per_state_losses)
    loss.backward()
    optimizer.step()
    current_loss = float(loss.data)
    losses.append(current_loss)
    return current_loss
