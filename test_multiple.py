import functools
import os
from typing import cast, Callable, List

import matplotlib.pyplot as plt
import torch
import torch as th

import data
import defaults
import disk_cache
from environment import Environment
import main as main_mod
from multi_nurbs_heliostat import MultiNURBSHeliostat
from render import Renderer
import training
import utils

join_paths = cast(
    Callable[[List[str]], str],
    functools.partial(functools.reduce, os.path.join),
)

# List of model paths that is created by joining path parts in each inner list.
MODEL_PATHS: List[str] = list(map(
    join_paths,
    [
        ['Results', 'Best10m', 'MakeRunableAgain_220410_1359', 'Logfiles', 'MultiNURBSHeliostat.pt'],
        ['Results', 'Best10m', 'MakeRunableAgain_220413_0947', 'Logfiles', 'MultiNURBSHeliostat.pt'],
    ],
))

# Use config from first model; used to create the environment and target
# heliostat.
CONFIG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(MODEL_PATHS[0])),
    'config.yaml',
)

SUN_DIRECTIONS = [
    [-0.43719268, 0.7004466, 0.564125],
    [-0.8662,  0.4890,  0.1026],
]

# height and width
BITMAP_DIMS = (512, 512)


def load_heliostat(
        path: str,
        device: th.device,
        receiver_center: torch.Tensor,
) -> MultiNURBSHeliostat:
    data = th.load(path, map_location=device)

    cached_from_dict = disk_cache.disk_cache(
        MultiNURBSHeliostat.from_dict,
        device,
        'cached',
        'test',
        ignore_argnums=[1],
    )

    heliostat = cached_from_dict(
        data,
        device,
        receiver_center=receiver_center,
    )
    return heliostat


def load_heliostats(
        paths: List[str],
        device: th.device,
        receiver_center: torch.Tensor,
) -> List[MultiNURBSHeliostat]:
    heliostats = [
        load_heliostat(path, device, receiver_center)
        for path in paths
    ]
    return heliostats


@th.no_grad()
def main() -> None:
    cfg = defaults.load_config_file(defaults.get_cfg_defaults(), CONFIG_PATH)

    if cfg.USE_FLOAT64:
        th.set_default_dtype(th.float64)
    else:
        th.set_default_dtype(th.float32)

    utils.fix_pytorch3d()

    th.manual_seed(cfg.SEED)
    device = th.device(
        'cuda'
        if cfg.USE_GPU and th.cuda.is_available()
        else 'cpu'
    )

    # Set up config from arguments
    bitmap_height, bitmap_width = BITMAP_DIMS
    cfg.merge_from_list([
        'SEED',
        cfg.SEED,
        'USE_GPU',
        cfg.USE_GPU,
        'AC.RECEIVER.RESOLUTION_X',
        bitmap_height,
        'AC.RECEIVER.RESOLUTION_Y',
        bitmap_width,

        'TEST.SUN_DIRECTIONS.CASE',
        'vecs',

        'TEST.SUN_DIRECTIONS.VECS.DIRECTIONS',
        SUN_DIRECTIONS,
    ])

    cached_build_target_heliostat = disk_cache.disk_cache(
        main_mod.build_target_heliostat,
        device,
        'cached',
        ignore_argnums=[1],
    )
    target_heliostat = cached_build_target_heliostat(cfg, device)
    env = Environment(cfg.AC, device)
    renderer = Renderer(target_heliostat, env)

    # Create targets
    cached_generate_sun_array = disk_cache.disk_cache(
        data.generate_sun_array,
        device,
        'cached',
        'test_',
        ignore_argnums=[1],
    )
    cached_generate_dataset = disk_cache.disk_cache(
        data.generate_dataset,
        device,
        'cached',
        'test',
        ignore_argnums=[3, 4, 5],
    )

    sun_directions, ae = cached_generate_sun_array(
        cfg.TEST.SUN_DIRECTIONS, device)
    targets = cached_generate_dataset(
        target_heliostat,
        env,
        sun_directions,
        None,
        'test',
        None,
    )

    # Set up heliostats
    heliostats = load_heliostats(MODEL_PATHS, device, cfg.AC.RECEIVER.CENTER)

    test_loss_func = training.build_loss_funcs(cfg.TRAIN.LOSS, [])[-1]

    for (i, (sun_direction, target)) in enumerate(zip(
            sun_directions,
            targets,
    )):
        for (j, heliostat) in enumerate(heliostats):
            heliostat_aligned = heliostat.align(sun_direction)

            pred_bitmap = cast(th.Tensor, renderer.render(heliostat_aligned))

            loss = test_loss_func(pred_bitmap, target)

            print('sun', i, 'h ' + str(j) + ':', loss.item())


if __name__ == '__main__':
    main()
