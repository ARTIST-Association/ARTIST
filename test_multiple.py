import functools
import os
from typing import cast

import matplotlib.pyplot as plt
import torch
import torch as th

import data
import defaults
from environment import Environment
import main as main_mod
from multi_nurbs_heliostat import MultiNURBSHeliostat
from render import Renderer
import training
import utils

# List of model paths that is created by joining path parts in each inner list.
MODEL_PATHS = list(map(
    functools.partial(functools.reduce, os.path.join),
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


class Target:
    __slots__ = [
        'heliostat_origin_center',
        'heliostat_face_normal',
        'heliostat_points',
        'heliostat_normals',
        'heliostat_up_dir',

        'receiver_origin_center',
        'receiver_width',
        'receiver_height',
        'receiver_normal',
        'receiver_up_dir',

        'sun',
        'num_rays',
        'mean',
        'cov',
        'xi',
        'yi',
    ]

    def __init__(self, checkpoint_dict):
        # We iterate the slots instead of the dictionary so missing data
        # is caught easily and redundant data is ignored.
        for key in self.__slots__:
            value = checkpoint_dict[key]
            setattr(self, key, value)


def _parse_bool(string):
    assert string == 'False' or string == 'True', \
        'please only use "False" or "True" as boolean arguments.'
    return string != 'False'


def load_heliostat(path, device, receiver_center):
    data = th.load(path, map_location=device)
    heliostat = MultiNURBSHeliostat.from_dict(
        data,
        device,
        receiver_center=receiver_center,
    )
    return heliostat


def load_heliostats(paths, device, receiver_center):
    heliostats = [
        load_heliostat(path, device, receiver_center)
        for path in paths
    ]
    return heliostats


def load_target(path, device):
    cp = th.load(path, map_location=device)
    target = Target(cp)
    return target


@th.no_grad()
def main():
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

    target_heliostat = main_mod.build_target_heliostat(cfg, device)
    env = Environment(cfg.AC, device)
    renderer = Renderer(target_heliostat, env)

    # Create targets
    sun_directions, ae = data.generate_sun_array(
        cfg.TEST.SUN_DIRECTIONS, device)
    targets = data.generate_dataset(
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

            pred_bitmap: torch.Tensor = cast(
                th.Tensor, renderer.render(heliostat_aligned))

            loss = test_loss_func(pred_bitmap, target)

            print('sun', i, 'h ' + str(j) + ':', loss.item())


if __name__ == '__main__':
    main()
