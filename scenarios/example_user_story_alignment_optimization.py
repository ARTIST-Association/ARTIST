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

warnings.filterwarnings("always")

# Attempt to import MPI.
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
    warnings.warn(
        "MPI is not available and distributed computing not possible. ARTIST will run on a single machine!",
        ImportWarning,
    )

# Set up MPI.
if MPI is not None:
    comm = MPI.COMM_WORLD
    world_size = comm.size
    rank = comm.rank
else:
    world_size = 1
    rank = 0


def test_alignment_optimization() -> None:

    torch.manual_seed(7)

    # aus calib file, aber welches?
    sun_azimuth = 
    sun_elevation =
    incident_ray_direction = utils.azimuth_elevation_to_enu(sun_azimuth, sun_elevation)

    # Load the scenario.
    # TODO
    # Sind die deflectometry files komplette scenarien oder nur defelc daten?
    # braucht man den scenario generator nicht mehr?
    # haben die Scenarien immer noch den selben aufbau?
    with h5py.File(f"{ARTIST_ROOT}/scenarios/test_alignment_optimization.h5", "r") as config_h5:
        scenario = Scenario.load_scenario_from_hdf5(scenario_file=config_h5)

    # TODO
    # Max fragen, wo in paint die Schwerpunkt funktion ist
    # Schwerpunkt aus Kalib-Bild extrahieren
    center_calibration_image = 

    optimizer = torch.optim.Adam([scenario.prototypes.kinematic.kinematic_deviations.first_joint_translation_e], lr=5e-3)

    for epoch in range(100):

        # Align heliostat.
        scenario.heliostats.heliostat_list[0].set_aligned_surface(
            incident_ray_direction=incident_ray_direction
        )

        # Create raytracer
        raytracer = HeliostatRayTracer(
            scenario=scenario, world_size=world_size, rank=rank, batch_size=10
        )

        final_bitmap = raytracer.trace_rays(incident_ray_direction=incident_ray_direction)

        # Apply all-reduce if MPI is used.
        if MPI is not None:
            final_bitmap = comm.allreduce(final_bitmap, op=MPI.SUM)
        final_bitmap = raytracer.normalize_bitmap(final_bitmap)

        # TODO
        # Schwerpunkt von final_bitmap berechnen
        center_ideal = 

        optimizer.zero_grad()

        loss = center_ideal - center_calibration_image
        loss.abs().mean().backward()

        optimizer.step()

        print(loss.abs().mean())