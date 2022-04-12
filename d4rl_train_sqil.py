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

import numpy as np
import gym
import pybullet_envs
import os
import pickle
import argparse

from datetime import datetime
from d4rl_pybullet.sac import SAC, seed_everything
from d4rl_pybullet.logger import SimpleLogger
from d4rl_pybullet.utility import save_buffer


def update(buffer, buffer_il, sac, batch_size, train_actor=True):
    obs_ts = []
    act_ts = []
    rew_tp1s = []
    obs_tp1s = []
    ter_tp1s = []
    while len(obs_ts) != batch_size:
        index = np.random.randint(len(buffer) - 1)
        # skip if index indicates the terminal state
        if buffer[index][3][0]:
            continue
        obs_ts.append(buffer[index][0])
        act_ts.append(buffer[index][1])
        rew_tp1s.append(buffer[index + 1][2])
        obs_tp1s.append(buffer[index + 1][0])
        ter_tp1s.append(buffer[index + 1][3])

    while len(obs_ts) != batch_size:
        index = np.random.randint(len(buffer) - 1)
        # skip if index indicates the terminal state
        if buffer_il[index][3][0]:
            continue
        obs_ts.append(buffer_il[index][0])
        act_ts.append(buffer_il[index][1])
        rew_tp1s.append(buffer_il[index + 1][2])
        obs_tp1s.append(buffer_il[index + 1][0])
        ter_tp1s.append(buffer_il[index + 1][3])

    critic_loss = sac.update_critic(obs_ts, act_ts, rew_tp1s, obs_tp1s, ter_tp1s)

    if train_actor:
        actor_loss = sac.update_actor(obs_ts)
    else:
        actor_loss = -1.0

    temp_loss = sac.update_temp(obs_ts)

    sac.update_target()

    return critic_loss, actor_loss, temp_loss


def evaluate(env, sac, n_episodes=10):
    episode_rews = []
    for episode in range(n_episodes):
        obs = env.reset()
        ter = False
        episode_rew = 0.0
        while not ter:
            act = sac.act([obs], deterministic=True)[0]
            obs, rew, ter, _ = env.step(act)
            episode_rew += rew
        episode_rews.append(episode_rew)
    return np.mean(episode_rews)


def train(
    env,
    eval_env,
    sac,
    logdir,
    desired_level,
    total_step,
    buffer_il=[],
    train_actor_threshold=None,
    batch_size=100,
    save_interval=10000,
    eval_interval=10000,
    recompute_reward=False,
    reward_recompute_interval=10000,
):
    logger = SimpleLogger(logdir)

    step = 0
    buffer = copy.deepcopy(buffer_il)
    while step <= total_step:
        obs_t = env.reset()
        ter_t = False
        rew_t = 0.0
        episode_rew = 0.0
        while not ter_t and step <= total_step:
            act_t = sac.act([obs_t])[0]

            buffer.append([obs_t, act_t, [rew_t], [ter_t]])

            obs_t, rew_t, ter_t, _ = env.step(act_t)

            episode_rew += rew_t

            if len(buffer) > batch_size:
                train_actor = (
                    step >= train_actor_threshold if train_actor_threshold else True
                )
                update(buffer, buffer_il, sac, batch_size, train_actor)

            # if step % save_interval == 0:
            #     sac.save(os.path.join(logdir, 'model_%d.pt' % step))

            if step % eval_interval == 0:
                logger.add("eval_reward", step, evaluate(eval_env, sac))

            step += 1

            if recompute_reward and step % reward_recompute_interval == 0:
                env.recompute_reward(buffer_il)

        if ter_t:
            buffer.append([obs_t, np.zeros_like(act_t), [rew_t], [ter_t]])

        logger.add("reward", step, episode_rew)

        if desired_level is not None and episode_rew >= desired_level:
            break

    # # save final buffer
    # save_buffer(buffer, logdir)
    # print('Final buffer has been saved.')
    #
    # # save final parameters
    # sac.save(os.path.join(logdir, 'final_model.pt'))
    # print('Final model parameters have been saved.')
