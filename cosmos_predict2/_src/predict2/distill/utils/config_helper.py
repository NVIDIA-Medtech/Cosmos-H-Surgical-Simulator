# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from cosmos_predict2._src.imaginaire.utils.checkpoint_db import get_checkpoint_path


def build_no_s3_run(
    job: dict,
    local_path: bool = False,
    resumable: bool = False,
    load_training_state: bool | None = None,
    wandb_mode: str = "offline",
) -> dict:
    """
    Make a copy of the input config that doesn't require S3 for checkpointing
    and I/O in the callbacks.

    ``resumable=True`` uses a stable output name across scheduler requeues.
    ``load_training_state`` controls only the initial load; subsequent local
    resumes still recover the complete training state.
    """
    # If local_path is True, use the local path as the load path
    if local_path:
        load_path = job["checkpoint"]["load_path"]
    else:
        model_url = f"s3://bucket/{job['checkpoint']['load_path']}/model"
        load_path = get_checkpoint_path(model_url)
    defaults = job.get("defaults", [])
    job_name = (
        f"{job['job']['name']}_no_s3_resumable"
        if resumable
        else f"{job['job']['name']}_no_s3" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"
    )
    if load_training_state is None:
        load_training_state = resumable
    no_s3_run = dict(
        defaults=defaults + ["_self_"] if "_self_" not in defaults else defaults,
        job=dict(
            name=job_name,
            wandb_mode=wandb_mode,
        ),
        checkpoint=dict(
            save_to_object_store=dict(enabled=False, credentials=""),
            load_from_object_store=dict(enabled=False),
            load_path=load_path,
            load_training_state=load_training_state,
        ),
        trainer=dict(
            straggler_detection=dict(enabled=False),
            callbacks=dict(
                heart_beat=dict(save_s3=False),
                iter_speed=dict(save_s3=False),
                device_monitor=dict(save_s3=False),
                every_n_sample_reg=dict(save_s3=False),
                every_n_sample_ema=dict(save_s3=False),
                wandb=dict(save_s3=False),
                wandb_10x=dict(save_s3=False),
                dataloader_speed=dict(save_s3=False),
            ),
        ),
    )
    return no_s3_run


def build_no_s3_run_v2(
    job: dict,
    local_path: bool = False,
    resumable: bool = False,
    load_training_state: bool | None = None,
    wandb_mode: str = "offline",
) -> dict:
    """Return a no-object-store run while preserving all source overrides.

    Unlike the legacy helper, this starts from a deep copy of the complete
    experiment. This is required by warmup and Self Forcing recipes whose
    action dimensions, checkpoint paths, and optimizer settings are supplied
    as nested experiment overrides.
    """
    from copy import deepcopy

    from omegaconf import OmegaConf

    try:
        job_dict = OmegaConf.to_container(job, resolve=False)
    except Exception:
        job_dict = dict(job)
    no_s3_run = deepcopy(job_dict)

    if local_path:
        load_path = job_dict["checkpoint"]["load_path"]
    else:
        model_url = f"s3://bucket/{job_dict['checkpoint']['load_path']}/model"
        load_path = get_checkpoint_path(model_url)

    job_name = (
        f"{job_dict['job']['name']}_no_s3_resumable"
        if resumable
        else f"{job_dict['job']['name']}_no_s3" + "_${now:%Y-%m-%d}_${now:%H-%M-%S}"
    )
    if load_training_state is None:
        load_training_state = resumable

    deep_update_config_dict(
        no_s3_run,
        dict(
            job=dict(name=job_name, wandb_mode=wandb_mode),
            checkpoint=dict(
                save_to_object_store=dict(enabled=False, credentials=""),
                load_from_object_store=dict(enabled=False),
                load_path=load_path,
                load_training_state=load_training_state,
            ),
            trainer=dict(
                straggler_detection=dict(enabled=False),
                callbacks=dict(
                    heart_beat=dict(save_s3=False),
                    iter_speed=dict(save_s3=False),
                    device_monitor=dict(save_s3=False),
                    every_n_sample_reg=dict(save_s3=False),
                    every_n_sample_ema=dict(save_s3=False),
                    wandb=dict(save_s3=False),
                    wandb_10x=dict(save_s3=False),
                    dataloader_speed=dict(save_s3=False),
                ),
            ),
        ),
    )

    defaults = no_s3_run.get("defaults", [])
    if "_self_" not in defaults:
        no_s3_run["defaults"] = defaults + ["_self_"]
    return no_s3_run


def deep_update_config_dict(dst: dict, src: dict) -> dict:
    """
    Updates nested dictionaries in the config dictionary (dst) with the values in src dictionary.
    Standard update in hydra only goes one level deep. This function goes arbitrarily deep.
    """
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            deep_update_config_dict(dst[k], v)
        else:
            dst[k] = v
    return dst
