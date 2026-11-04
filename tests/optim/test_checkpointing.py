import copy
import pathlib

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.optim.aim_point_optimizer import AimPointOptimizer
from artist.scenario.scenario import Scenario
from artist.util import constants

SCENARIO = pathlib.Path(ARTIST_ROOT) / "tests/data/scenarios/test_scenario_paint_four_heliostats.h5"


def _make_optimizer(ddp_setup, device, batch_size, checkpointing):
    with h5py.File(SCENARIO, "r") as f:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file=f, device=device)
    config = {
        constants.optimization: {
            constants.initial_learning_rate: 0.03, 
            constants.tolerance: 0.0,
            constants.max_epoch: 1, 
            constants.batch_size: batch_size, 
            constants.log_step: 1,
            constants.early_stopping_delta: 1.0, 
            constants.early_stopping_patience: 2,
            constants.early_stopping_window: 2,
            "gradient_checkpointing": checkpointing,
        },
        constants.scheduler: {
            constants.scheduler_type: "reduce_on_plateau", 
            constants.lr_min: 1e-06,
            constants.reduce_factor: 0.2,
            constants.patience: 10,
            constants.threshold: 0.001,
            constants.cooldown: 5
        },
        constants.constraints: {
            "lambda_intensity_reward": 0.1,
            constants.rho_flux_integral: 1000.0,
            constants.rho_local_flux: 10.0,
            constants.max_flux_density: 1000000
        },
    }
    path = pathlib.Path(ARTIST_ROOT) / "tests/data/expected_test_data.pt"
    expected_key = "ground_truth_flux"
    ground_truth = torch.load(path, map_location=device, weights_only=True)[
        expected_key
    ]
    return AimPointOptimizer(
        ddp_setup=ddp_setup, 
        scenario=scenario, 
        optimization_configuration=config,
        incident_ray_direction=torch.tensor([0.0, 1.0, 0.0, 0.0], device=device),
        target_area_index=3, 
        ground_truth=ground_truth,
        dni=800,
        data_dir=pathlib.Path("None"),
        bitmap_resolution=torch.tensor([256, 256], device=device), 
        device=device,
    )


def _flux_and_grads(opt, device):
    """One forward + backward exactly as in optimize(), returns flux and parameter grads."""
    (params, scales, init_mp, masks, targets, incident, offsets) = opt._initialize_group_parameters(device)
    with torch.no_grad():
        for p in params:
            p.normal_(0.0, 0.002, generator=torch.Generator(device=device).manual_seed(0))
    optimizer, _, _ = opt._setup_optimizer_scheduler_early_stopping(params)
    opt._shading_caches = {}
    opt._align_all_groups(optimizer, init_mp, scales, masks, device)
    flux, _ = opt._trace_and_accumulate_flux(masks, targets, incident, offsets, device)
    weights = torch.linspace(0, 1, flux.numel(), device=device).view_as(flux)
    (flux * weights).square().sum().backward()
    return flux.detach(), [p.grad.detach().clone() for p in params]


@pytest.mark.parametrize("batch_size", [1, 2])
def test_checkpointing_matches_no_checkpointing(ddp_setup_for_testing, device, batch_size):
    ddp_setup_for_testing["device"] = device
    ddp_setup_for_testing["groups_to_ranks_mapping"] = {0: [0, 1]}

    ref_flux, ref_grads = _flux_and_grads(
        _make_optimizer(copy.deepcopy(ddp_setup_for_testing), device, 50, checkpointing=False), device)
    flux, grads = _flux_and_grads(
        _make_optimizer(copy.deepcopy(ddp_setup_for_testing), device, batch_size, checkpointing=True), device)

    # scatter_add_ on GPU is non-deterministic in summation order -> small tolerance.
    torch.testing.assert_close(flux, ref_flux, rtol=1e-5, atol=1e-6 * ref_flux.abs().max())
    for g, g_ref in zip(grads, ref_grads):
        assert g_ref.abs().sum() > 0, "Test not working because reference gradients are 0."
        torch.testing.assert_close(g, g_ref, rtol=1e-4, atol=1e-6 * g_ref.abs().max())