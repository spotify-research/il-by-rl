from functools import partial
from utils import execute, env_paths
from multiprocessing import Pool

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

import argparse

from train_bc_functions import train_bc_actor
from train_il_functions import train_il_actor
from train_gail_functions import train_gail_actor
from train_gmmil_functions import train_gmmil_actor
from train_sqil_functions import train_sqil_actor


def run_experiments(train_function, env, method_name, total_steps):
    env_name, h5path = env_paths[env]
    tasks = []
    for seed in [1, 2, 3, 4, 5]:
        for ep in [1, 2, 4, 8, 16]:
            tasks.append(
                partial(
                    train_function,
                    no_episodes=ep,
                    env_name=env_name,
                    seed=seed,
                    h5path=h5path,
                    total_steps=total_steps,
                    file_path=f"{env_name}/{method_name}_model_ep{ep}_seed{seed}",
                )
            )

        for ep in [2, 4, 8, 16, 32, 64, 128, 256, 512]:
            tasks.append(
                partial(
                    train_function,
                    no_episodes=1 / ep,
                    env_name=env_name,
                    seed=seed,
                    h5path=h5path,
                    total_steps=total_steps,
                    file_path=f"{env_name}/{method_name}_model_ep1_{ep}_seed{seed}",
                )
            )

    with Pool(processes=60) as pool:
        actors = list(pool.map(execute, tasks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env", type=str, default="hopper")
    parser.add_argument("--method", type=str, default="bc")
    parser.add_argument("--steps", type=int, default=500000)
    args = parser.parse_args()
    train_functions_dict = {
        "bc": train_bc_actor,
        "il": train_il_actor,
        "gail": train_gail_actor,
        "gmmil": train_gmmil_actor,
        "sqil": train_sqil_actor,
    }

    run_experiments(
        train_functions_dict[args.method], args.env, args.method, args.steps
    )


if __name__ == "__main__":
    main()
