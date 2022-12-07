import torch
import torch as th
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode
import functools
import disk_cache
import data
from heliostat_models import AbstractHeliostat, Heliostat
from environment import Environment
from typing import (
    Any,
    Callable,
    cast,
    Dict,
    List,
    Optional,
    Tuple,
    Type,
    Union,
)
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
    def make_cached_generate_sun_array(
            prefix: str = '',
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

    def make_cached_generate_dataset(
            prefix: str,
            tb_log: bool = True,
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
    def make_cached_generate_sun_array(
            prefix: str = '',
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

    def make_cached_generate_dataset(
            prefix: str,
            tb_log: bool = True,
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

    return (
        (
        make_cached_generate_sun_array('grid_'),
        make_cached_generate_sun_array('spheric_'),
        make_cached_generate_sun_array('season_'),
        ),
        (
        make_cached_generate_dataset('grid', False),
        make_cached_generate_dataset('naive_grid', False),
        make_cached_generate_dataset('spheric', False),
        make_cached_generate_dataset('naive_spheric', False),
        make_cached_generate_dataset('season', False),
        make_cached_generate_dataset('naive_season', False),
        ),
    )