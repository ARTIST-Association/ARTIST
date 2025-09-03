import logging
import pathlib

import h5py
import torch
from matplotlib import pyplot as plt

from artist.core import loss_functions
from artist.core.regularizers import IdealSurfaceRegularizer, TotalVariationRegularizer
from artist.core.surface_reconstructor import SurfaceReconstructor
from artist.data_loader import paint_loader
from artist.scenario.scenario import Scenario
from artist.util import config_dictionary, set_logger_config, utils
from artist.util.environment_setup import get_device
from artist.util.nurbs import NURBSSurfaces

torch.manual_seed(7)
torch.cuda.manual_seed(7)


def plot_normal_angle_map(surface_points, surface_normals, reference_direction, name):  # noqa: D103
    plt.clf()
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(15, 6))
    normals = (
        (
            surface_normals[..., :3]
            / torch.linalg.norm(surface_normals[..., :3], axis=-1, keepdims=True)
        )
        .cpu()
        .detach()
    )
    ref = (
        (reference_direction[..., :3] / torch.linalg.norm(reference_direction[..., :3]))
        .cpu()
        .detach()
    )

    for facet_points, facet_normals in zip(surface_points.cpu().detach(), normals):
        x, y, z = (
            facet_points[:, 0].cpu().detach(),
            facet_points[:, 1].cpu().detach(),
            facet_points[:, 2].cpu().detach(),
        )

        sc0 = axes[0].scatter(x, y, c=z, cmap="viridis", s=7)  # noqa: F841
        axes[0].set_title("Surface points")

        cos_theta = facet_normals @ ref
        angles = torch.arccos(torch.clip(cos_theta, -1.0, 1.0))

        angles = torch.clip(angles, -0.1, 0.1)

        sc1 = axes[1].scatter(x, y, c=angles, cmap="plasma", s=7)  # noqa: F841
        axes[1].set_title("Angle map normals")

    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.clf()
    plt.close()


def plot_multiple_fluxes(reconstructed, references, name):  # noqa: D103
    fig1, axes1 = plt.subplots(nrows=reconstructed.shape[0], ncols=2, figsize=(24, 150))
    for i in range(reconstructed.shape[0]):
        axes1[i, 0].imshow(reconstructed[i].cpu().detach(), cmap="gray")
        axes1[i, 0].set_title(f"Reconstructed {i}")
        axes1[i, 0].axis("off")

        axes1[i, 1].imshow(references[i].cpu().detach(), cmap="gray")
        axes1[i, 1].set_title(f"Reference {i}")
        axes1[i, 1].axis("off")

    plt.tight_layout()
    plt.savefig(f"{name}.png")
    plt.clf()
    plt.close()


#####################################################

# Set up logger
set_logger_config()
log = logging.getLogger(__name__)

# Set the device
device = get_device()

params = {
    "number_of_surface_points": 47,
    "number_of_control_points": 61,
    "number_of_rays": 185,
    "number_of_training_samples": 3,
    "nurbs_degree": 2,
    "scheduler": "cyclic",
    "lr_gamma": 8.90e-1,
    "lr_min": 1.89e-5,
    "lr_max": 1.00e-3,
    "lr_step_size_up": 716,
    "lr_reduce_factor": 2.87e-1,
    "lr_patience": 43,
    "lr_threshold": 3.10e-4,
    "lr_cooldown": 0,
    "ideal_surface_loss_weight": 7.48e-2,
    "total_variation_loss_points_weight": 9.46e-1,
    "total_variation_loss_normals_weight": 3.11e-1,
    "total_variation_loss_number_of_neighbors_points": 395,
    "total_variation_loss_sigma_points": 0,
    "total_variation_loss_number_of_neighbors_normals": 1379,
    "total_variation_loss_sigma_normals": 0,
    "initial_learning_rate": 3.56e-1,
    "loss_class": "KLDivergenceLoss",
}

# loss 1.62E-1, island 0, worker 3, generation 125]

ddp_setup = {
    config_dictionary.device: device,
    config_dictionary.is_distributed: False,
    config_dictionary.is_nested: False,
    config_dictionary.rank: 0,
    config_dictionary.world_size: 1,
    config_dictionary.process_subgroup: None,
    config_dictionary.groups_to_ranks_mapping: {0: [0]},
    config_dictionary.heliostat_group_rank: 0,
    config_dictionary.heliostat_group_world_size: 1,
    config_dictionary.ranks_to_groups_mapping: {0: [0]},
}


number_of_surface_points_per_facet = torch.tensor(
    [params["number_of_surface_points"], params["number_of_surface_points"]],
    device=device,
)

number_of_control_points_per_facet = torch.tensor(
    [params["number_of_control_points"], params["number_of_control_points"]],
    device=device,
)


with h5py.File(
    pathlib.Path(
        "/workVERLEIHNIX/mb/ARTIST/examples/hyperparameter_optimization/data_to_be_removed/scenario.h5"
    ),
    "r",
) as scenario_file:
    scenario = Scenario.load_scenario_from_hdf5(
        scenario_file=scenario_file,
        number_of_surface_points_per_facet=number_of_surface_points_per_facet,
        change_number_of_control_points_per_facet=number_of_control_points_per_facet,
        device=device,
    )

scenario.light_sources.light_source_list[0].number_of_rays = int(
    params["number_of_rays"]
)

for heliostat_group in scenario.heliostat_field.heliostat_groups:
    heliostat_group.nurbs_degrees = torch.tensor(
        [params["nurbs_degree"], params["nurbs_degree"]], device=device
    )

heliostat_data_mapping = paint_loader.build_heliostat_data_mapping(
    base_path="/workVERLEIHNIX/mb/ARTIST/examples/hyperparameter_optimization/data_to_be_removed/paint",
    heliostat_names=["AA39"],
    number_of_measurements=int(params["number_of_training_samples"]),
    image_variant="flux-centered",
    randomize=False,
)

data: dict[str, str | list[tuple[str, list[pathlib.Path], list[pathlib.Path]]]] = {
    config_dictionary.data_source: config_dictionary.paint,
    config_dictionary.heliostat_data_mapping: heliostat_data_mapping,
}

scheduler = params["scheduler"]
scheduler_parameters = {
    config_dictionary.gamma: params["lr_gamma"],
    config_dictionary.min: params["lr_min"],
    config_dictionary.max: params["lr_max"],
    config_dictionary.step_size_up: params["lr_step_size_up"],
    config_dictionary.reduce_factor: params["lr_reduce_factor"],
    config_dictionary.patience: params["lr_patience"],
    config_dictionary.threshold: params["lr_threshold"],
    config_dictionary.cooldown: params["lr_cooldown"],
}

ideal_surface_regularizer = IdealSurfaceRegularizer(
    weight=params["ideal_surface_loss_weight"], reduction_dimensions=(1, 2, 3, 4)
)
total_variation_regularizer_points = TotalVariationRegularizer(
    weight=params["total_variation_loss_points_weight"],
    reduction_dimensions=(1,),
    surface=config_dictionary.surface_points,
    number_of_neighbors=int(params["total_variation_loss_number_of_neighbors_points"]),
    sigma=params["total_variation_loss_sigma_points"],
)
total_variation_regularizer_normals = TotalVariationRegularizer(
    weight=params["total_variation_loss_normals_weight"],
    reduction_dimensions=(1,),
    surface=config_dictionary.surface_normals,
    number_of_neighbors=int(params["total_variation_loss_number_of_neighbors_normals"]),
    sigma=params["total_variation_loss_sigma_normals"],
)

regularizers = [
    ideal_surface_regularizer,
    total_variation_regularizer_points,
    total_variation_regularizer_normals,
]

optimization_configuration = {
    config_dictionary.initial_learning_rate: params["initial_learning_rate"],
    config_dictionary.tolerance: 0.00005,
    config_dictionary.max_epoch: 3000,
    config_dictionary.num_log: 300,
    config_dictionary.early_stopping_delta: 1e-4,
    config_dictionary.early_stopping_patience: 20,
    config_dictionary.scheduler: scheduler,
    config_dictionary.scheduler_parameters: scheduler_parameters,
    config_dictionary.regularizers: regularizers,
}

surface_reconstructor = SurfaceReconstructor(
    ddp_setup=ddp_setup,
    scenario=scenario,
    data=data,
    optimization_configuration=optimization_configuration,
    number_of_surface_points=number_of_surface_points_per_facet,
    bitmap_resolution=torch.tensor([256, 256], device=device),
    device=device,
)

loss_class = getattr(loss_functions, str(params["loss_class"]))
loss_definition = (
    loss_functions.PixelLoss(scenario=scenario)
    if loss_class is loss_functions.PixelLoss
    else loss_functions.KLDivergenceLoss()
)

loss = surface_reconstructor.reconstruct_surfaces(
    loss_definition=loss_definition,
    device=device,
)

print(f"loss: {loss}")

for index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
    torch.save(
        heliostat_group.nurbs_control_points,
        f"/workVERLEIHNIX/mb/ARTIST/examples/hyperparameter_optimization/results/group_{index}_cp.pt",
    )

evaluation_points = (
    utils.create_nurbs_evaluation_grid(
        number_of_evaluation_points=number_of_surface_points_per_facet,
        device=device,
    )
    .unsqueeze(0)
    .unsqueeze(0)
    .expand(
        heliostat_group.number_of_active_heliostats,
        heliostat_group.number_of_facets_per_heliostat,
        -1,
        -1,
    )
)
for index, heliostat_group in enumerate(scenario.heliostat_field.heliostat_groups):
    for i in range(heliostat_group.number_of_heliostats):
        temp_nurbs = NURBSSurfaces(
            degrees=heliostat_group.nurbs_degrees,
            control_points=heliostat_group.nurbs_control_points[index].unsqueeze(0),
            device=device,
        )

        temp_points, temp_normals = temp_nurbs.calculate_surface_points_and_normals(
            evaluation_points=evaluation_points[0].unsqueeze(0),
            device=device,
        )

        name1 = pathlib.Path(
            f"/workVERLEIHNIX/mb/ARTIST/examples/hyperparameter_optimization/results/points_and_normals_h_{i}"
        )

        plot_normal_angle_map(
            surface_points=temp_points[0],
            surface_normals=temp_normals[0],
            reference_direction=torch.tensor([0.0, 0.0, 1.0, 0.0], device=device),
            name=name1,
        )
