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


def make_cached_generate_sun_array_factory(
        device: th.device,
) -> Callable[
    [str],
    Callable[
        [CfgNode, th.device, Optional[torch.Tensor], Optional[str]],
        Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, Any]]],
    ],
]:
    def make_cached_generate_sun_array(
            prefix: str,
    ) -> Callable[
        [CfgNode, th.device, Optional[torch.Tensor], Optional[str]],
        Tuple[torch.Tensor, Union[torch.Tensor, Dict[str, Any]]],
    ]:
        return disk_cache.disk_cache(
            data.generate_sun_array,
            device,
            'cached',
            prefix,
            ignore_argnums=[1],
        )

    return make_cached_generate_sun_array


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


def set_up_dataset_caching(
        device: th.device,
        writer: Optional[SummaryWriter],
) -> Tuple[
    Tuple[Callable, Callable],
    Tuple[
        Callable,
        Callable,
        Callable,
    ],
]:
    make_cached_generate_sun_array = make_cached_generate_sun_array_factory(
        device)
    make_cached_generate_dataset = make_cached_generate_dataset_factory(
        device, writer)

    return (
        (
            make_cached_generate_sun_array('target_'),
            make_cached_generate_sun_array('test_'),
        ),
        (
            make_cached_generate_dataset('train'),
            make_cached_generate_dataset('pretrain'),
            make_cached_generate_dataset('test'),
        ),
    )


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
    make_cached_generate_sun_array = make_cached_generate_sun_array_factory(
        device)
    make_cached_generate_dataset = make_cached_generate_dataset_factory(
        device, writer, False)

    return (
        (
            make_cached_generate_sun_array('grid_'),
            make_cached_generate_sun_array('spheric_'),
            make_cached_generate_sun_array('season_'),
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
