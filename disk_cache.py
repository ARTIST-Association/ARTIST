import functools
import glob
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TypeVar

import torch
import torch as th
from torch.utils.tensorboard import SummaryWriter

from environment import Environment, Sun_Distribution
from heliostat_models import AbstractHeliostat

R = TypeVar('R')
F = Callable[..., R]


class ExtendedEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, th.device):
            return 0
        elif isinstance(obj, AbstractHeliostat):
            discrete_points = obj.discrete_points
            normals = obj.normals
            return (discrete_points, normals)
        elif isinstance(obj, th.Tensor):
            return obj.tolist()
        elif isinstance(obj, Environment):
            attrs = vars(obj).copy()
            del attrs['cfg']
            return attrs
        elif isinstance(obj, Sun_Distribution):
            return (obj.cfg, obj.num_rays)
        elif isinstance(obj, SummaryWriter):
            return None

        return super().default(obj)


def find_disk_hash(
        save_dir: Path,
        file_prefix: str,
) -> Tuple[Optional[str], Optional[Path]]:
    paths = save_dir.glob(glob.escape(file_prefix) + '*.pt')
    least_recent = min(
        paths,
        key=lambda path: path.stat().st_mtime,
        default=None,
    )
    if least_recent is None:
        return None, None
    hash_value = least_recent.name[len(file_prefix):-3]
    return hash_value, least_recent


def hash_args(
        func: Callable,
        args: Tuple,
        kwargs: Dict[str, Any],
        ignore_argnums: List = [],
        ignore_argnames: List = [],
) -> str:
    # Set up argument handling.
    sig = inspect.signature(func)
    all_args = list(sig.parameters.values())

    prev_argnames = ignore_argnames.copy()
    ignore_argnames.extend((
        all_args[ignore_index].name
        for ignore_index in ignore_argnums
    ))
    ignore_argnums.extend((
        i
        for (i, name) in enumerate(map(
                lambda arg: arg.name,
                all_args,
        ))
        if name in prev_argnames
    ))

    # Find arguments we want to hash.
    hash_args = tuple(
        arg
        for (i, arg) in enumerate(args)
        if i not in ignore_argnums
    )
    hash_kwargs = tuple(
        key
        for key in kwargs.keys()
        if key not in ignore_argnames
    )

    # Hash arguments and RNG state.
    hash_val = hashlib.md5(json.dumps(
        (hash_args, hash_kwargs, th.get_rng_state(), th.cuda.get_rng_state()),
        sort_keys=True,
        cls=ExtendedEncoder,
    ).encode()).hexdigest()
    return hash_val


@th.no_grad()
def disk_cache(
        func: F,
        device: th.device,
        save_dir: str,
        prefix: str = '',
        ignore_argnums: List = [],
        ignore_argnames: List = [],
        on_load: Optional[Callable[[R], R]] = None,
        remove_outdated: bool = False,
) -> F:
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        file_prefix = prefix + func.__name__ + '_'

        hash_val = hash_args(
            func,
            args,
            kwargs,
            ignore_argnums,
            ignore_argnames,
        )
        new_path = save_dir / (file_prefix + hash_val + '.pt')

        if new_path.is_file():
            result, rng_states = th.load(new_path, map_location=device)
            # Place RNG states on CPU.
            rng_states = (
                rng_states[0].cpu(),
                (
                    [t.cpu() for t in rng_states[1]]
                    if rng_states[1] is not None
                    else None
                ),
            )

            if on_load:
                result = on_load(result)
            th.set_rng_state(rng_states[0])
            if th.cuda.is_available() and rng_states[1] is not None:
                th.cuda.set_rng_state_all(rng_states[1])
        else:
            result = func(*args, **kwargs)

            if remove_outdated:
                prev_hash_val, prev_path = find_disk_hash(
                    save_dir, file_prefix)

            th.save((
                result,
                (
                    th.get_rng_state(),
                    (
                        th.cuda.get_rng_state_all()
                        if th.cuda.is_available()
                        else None
                    ),
                ),
            ), new_path)

            if remove_outdated and prev_path is not None:
                prev_path.unlink()
        return result

    return wrapped
