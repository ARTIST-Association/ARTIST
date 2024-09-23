import json
import warnings

import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.field.kinematic_rigid_body import RigidBody
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.scenario import Scenario
from artist.util import utils
from artist.util import config_dictionary
from artist.util.configuration_classes import (
    ActuatorParameters,
    ActuatorConfig,
    ActuatorListConfig,
    KinematicDeviations,
)

import matplotlib.pyplot as plt

def test_alignment_optimization() -> None:

    torch.manual_seed(7)

    calibration_item_stac_file = f"{ARTIST_ROOT}/measurement_data/AA39/Calibration/86500-calibration-item-stac.json"
    with open(calibration_item_stac_file, 'r') as file:
        calibration_dict = json.load(file)
        sun_azimuth = torch.tensor(calibration_dict["view:sun_azimuth"])
        sun_elevation = torch.tensor(calibration_dict["view:sun_elevation"])

    calibration_properties_file = f"{ARTIST_ROOT}/measurement_data/AA39/Calibration/86500-calibration-properties.json"
    with open(calibration_properties_file, 'r') as file:
        calibration_dict = json.load(file)
        center_calibration_image = utils.convert_3d_points_to_4d_format(torch.tensor(calibration_dict["focal_spot"]["UTIS"]))
    
    incident_ray_direction = utils.convert_3d_direction_to_4d_format(utils.azimuth_elevation_to_enu(sun_azimuth, sun_elevation, degree=False))

    # Load the scenario.
    with h5py.File(f"{ARTIST_ROOT}/scenarios/test_alignment_optimization.h5", "r") as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file=config_h5)

    optimizer = torch.optim.Adam([scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_e], lr=5e-3)

    for epoch in range(100):

        # Align heliostat.
        scenario.heliostats.heliostat_list[0].set_aligned_surface(
            incident_ray_direction=incident_ray_direction
        )

        # Create raytracer
        raytracer = HeliostatRayTracer(
            scenario=scenario
        )

        final_bitmap = raytracer.trace_rays(incident_ray_direction=incident_ray_direction)
        final_bitmap = raytracer.normalize_bitmap(final_bitmap)

        plt.imshow(final_bitmap.detach().numpy())
        plt.savefig("bitmap.png")


        # TODO
        # Schwerpunkt von final_bitmap berechnen
        center_ideal = 0

        optimizer.zero_grad()

        loss = center_ideal - center_calibration_image
        loss.abs().mean().backward()

        optimizer.step()

        print(loss.abs().mean())