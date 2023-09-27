import functools
from typing import (
    Any,
    Callable,
    Dict,
    Optional,
    Tuple,
    Union,
)

import torch
import torch as th
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode

import data
import disk_cache
from environment import Environment
from heliostat_models import AbstractHeliostat


def make_cached_generate_light_array_factory(
        device: th.device,
) -> Callable[
                [str],
                Callable[
                    [CfgNode, th.device, Optional[torch.Tensor], Optional[str]],
                    Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, Any]]],
                ],
]:
    def make_cached_generate_light_array(
            prefix: str,
    ) -> Callable[
        [CfgNode, th.device, Optional[torch.Tensor], Optional[str]],
        Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, Any]]],
    ]:
        return disk_cache.disk_cache(
            data.generate_light_array,
            device,
            'cached',
            prefix,
            ignore_argnums=[1],
        )

    return make_cached_generate_light_array


def make_cached_generate_dataset_factory(
        device: th.device,
        writer: Optional[SummaryWriter],
        tb_log: bool = True,
) -> Callable[
    [str],
    Callable[
        [
            AbstractHeliostat,
            Environment,
            torch.Tensor,
            Optional[str],
            str,
            Optional[SummaryWriter],
        ],
        torch.Tensor,
    ],
]:
    def make_cached_generate_dataset(
            prefix: str,
    ) -> Callable[
        [
            AbstractHeliostat,
            Environment,
            torch.Tensor,
            Optional[str],
            str,
            Optional[SummaryWriter],
        ],
        torch.Tensor,
    ]:
        if tb_log:
            log_dataset: Optional[disk_cache.OnLoadFn] = functools.partial(
                data.log_dataset,
                prefix,
                writer,
            )
        else:
            log_dataset = None

        return disk_cache.disk_cache(
            data.generate_dataset,
            device,
            'cached',
            prefix,
            on_load=log_dataset,
            ignore_argnums=[3, 4, 5],
        )

    return make_cached_generate_dataset


def set_up_light_position_caching(
        device: th.device,
        train_or_test: str,
) -> Tuple[Callable, Callable]:
    make_cached_generate_light_array = make_cached_generate_light_array_factory(
        device)
    if train_or_test=="train":
        return make_cached_generate_light_array('train_')
    elif train_or_test=="test":
        return make_cached_generate_light_array('test_')
    else:
        raise ValueError("Invalid option for train_or_test. use string \"train\" or \"test\".")
    

def set_up_dataset_caching(
        device: th.device,
        writer: Optional[SummaryWriter],
        which_dataset: str,
) -> Tuple[Callable,Callable,Callable]:
    make_cached_generate_dataset = make_cached_generate_dataset_factory(
        device, writer)

    if which_dataset == "train":
        return make_cached_generate_dataset('train')
    elif which_dataset == "test":
        return make_cached_generate_dataset('test')
    elif which_dataset == "pretrain":
        return make_cached_generate_dataset('pretrain')
    else:
        raise ValueError("Invalid option for which_dataset use string \"train\",  \"test\" or \"pretrain\".")
    

def set_up_test_dataset_caching(
        device: th.device,
        writer: Optional[SummaryWriter],
) -> Tuple[
    Tuple[Callable, Callable, Callable],
    Tuple[
        Callable,
        Callable,
        Callable,
        Callable,
        Callable,
        Callable,
    ],
]:
    make_cached_generate_light_array = make_cached_generate_light_array_factory(
        device)
    make_cached_generate_dataset = make_cached_generate_dataset_factory(
        device, writer, False)

    return (
        (
            make_cached_generate_light_array('grid_'),
            make_cached_generate_light_array('spheric_'),
            make_cached_generate_light_array('season_'),
        ),
        (
            make_cached_generate_dataset('grid'),
            make_cached_generate_dataset('naive_grid'),
            make_cached_generate_dataset('spheric'),
            make_cached_generate_dataset('naive_spheric'),
            make_cached_generate_dataset('season'),
            make_cached_generate_dataset('naive_season'),
        ),
    )
