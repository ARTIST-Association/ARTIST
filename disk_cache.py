import functools
import glob
import hashlib
import inspect
import json
from pathlib import Path
from typing import (
    Any,
    Callable,
    Dict,
    Iterator,
    List,
    Optional,
    Tuple,
    TypeVar,
)

import torch
import torch as th
from torch.utils.tensorboard import SummaryWriter
from yacs.config import CfgNode

from environment import Environment, Sun_Distribution
from heliostat_models import AbstractHeliostat

R = TypeVar('R')
A = TypeVar('A')
F = Callable[..., R]
OnLoadFn = Callable[[R], R]

DISABLE_DISK_CACHE = False

# This magic number should be updated to invalidate out-of-date caches
# due to breaking changes.
_CACHE_VERSION = 6

_DYNAMIC_CONFIG_KEYS = ['LOGDIR']


class ExtendedEncoder(json.JSONEncoder):
    def default(self, obj: Any) -> Any:
        if isinstance(obj, th.device):
            return 0
        elif isinstance(obj, AbstractHeliostat):
            return (
                obj.discrete_points,
                obj.normals,
                obj.position_on_field,
                obj.aim_point,
                obj.focus_point,
                obj.disturbance_angles,
            )
        elif isinstance(obj, th.Tensor):
            return (obj.tolist(), obj.dtype)
        elif isinstance(obj, th.dtype):
            return repr(obj)
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


def _normalize_configs(
        hash_args: Iterator[Any],
        hash_kwargs: Iterator[Tuple[str, Any]],
) -> Tuple[List[Any], List[Tuple[str, Any]]]:
    # Remove changing config values.
    new_hash_args = []
    for arg in hash_args:
        if isinstance(arg, CfgNode):
            arg = arg.clone()
            for key in _DYNAMIC_CONFIG_KEYS:
                arg.pop(key, None)
        new_hash_args.append(arg)

    new_hash_kwargs: List[Tuple[str, Any]] = []
    for (name, arg) in hash_kwargs:
        if isinstance(arg, CfgNode):
            arg = arg.clone()
            for key in _DYNAMIC_CONFIG_KEYS:
                arg.pop(key, None)
        new_hash_args.append((name, arg))

    return new_hash_args, new_hash_kwargs


def _get_indentation(line: str) -> int:
    indentation = 0
    char = line[indentation]
    while char == ' ':
        indentation += 1
        char = line[indentation]
    return indentation


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

    try:
        code_str = inspect.getsource(func)
        bytecode: Optional[List] = list(
            compile(code_str, '<string>', 'exec').co_code)
    except IndentationError:
        # Handle functions inside a class; we just try to remove space
        # indentation.
        lines = code_str.split('\n')
        indentation = _get_indentation(lines[0])

        code_str = '\n'.join(map(lambda l: l[indentation:], lines))
        bytecode = list(
            compile(code_str.lstrip(), '<string>', 'exec').co_code)
    except OSError:
        bytecode = None

    # Find arguments we want to hash.
    hash_args = (
        arg
        for (i, arg) in enumerate(args)
        if i not in ignore_argnums
    )
    hash_kwargs = (
        (key, value)
        for (key, value) in kwargs.items()
        if key not in ignore_argnames
    )

    new_hash_args, new_hash_kwargs = _normalize_configs(hash_args, hash_kwargs)

    try:
        # Hash arguments and RNG state.
        hash_val = hashlib.md5(json.dumps(
            (
                new_hash_args,
                new_hash_kwargs,
                bytecode,
                _CACHE_VERSION,
                th.get_rng_state(),
                (
                    th.cuda.get_rng_state_all()
                    if th.cuda.is_available()
                    else None
                ),
            ),
            sort_keys=True,
            cls=ExtendedEncoder,
        ).encode()).hexdigest()
    except Exception as e:
        print(
            'The following error was encountered during disk cache hashing. '
            'Please try setting `disk_cache.DISABLE_DISK_CACHE = True` or '
            'delete your cache directory.'
        )
        raise e
    return hash_val


@th.no_grad()
def disk_cache(
        func: F,
        device: th.device,
        save_dir: str,
        prefix: str = '',
        ignore_argnums: List = [],
        ignore_argnames: List = [],
        on_load: Optional[OnLoadFn] = None,
        remove_outdated: bool = False,
) -> F:
    if DISABLE_DISK_CACHE:
        return func

    save_dir: Path = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    @functools.wraps(func)
    def wrapped(*args: Any, **kwargs: Any) -> R:
        assert isinstance(save_dir, Path)
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
