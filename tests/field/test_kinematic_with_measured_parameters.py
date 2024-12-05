import pathlib
from artist import ARTIST_ROOT
from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.scenario import Scenario
import h5py
import pytest
import torch

@pytest.fixture(params=["cpu", "cuda:1"] if torch.cuda.is_available() else ["cpu"])
def device(request: pytest.FixtureRequest) -> torch.device:
    """
    Return the device on which to initialize tensors.

    Parameters
    ----------
    request : pytest.FixtureRequest
        The pytest fixture used to consider different test cases.

    Returns
    -------
    torch.device
        The device on which to initialize tensors.
    """
    return torch.device(request.param)

def test_kinematic(
    device: torch.device,
) -> None:
    """
    Test the alignemnt optimization methods.

    Parameters
    ----------
    device : torch.device
        The device on which to initialize tensors.

    Raises
    ------
    AssertionError
        If test does not complete as expected.
    """
    torch.manual_seed(7)
    torch.cuda.manual_seed(7)

    scenario_path = pathlib.Path(ARTIST_ROOT) / f"tests/data/test_scenario_paint.h5"
    with h5py.File(scenario_path, "r") as scenario_file:
        scenario = Scenario.load_scenario_from_hdf5(
            scenario_file=scenario_file, device=device
        )

    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_e=torch.tensor(0.0, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_n=torch.tensor(0.0, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_u=torch.tensor(0.0, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_e=torch.tensor(0.0, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_n=torch.tensor(0.0, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_u=torch.tensor(0.0, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_e=torch.tensor(0.0, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_n=torch.tensor(0.0, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_u=torch.tensor(0.315, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_e=torch.tensor(0.0, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_n=torch.tensor(0.0, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_u=torch.tensor(0.0, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_e=torch.tensor(0.0, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_n=torch.tensor(-0.17755, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_u=torch.tensor(-0.4045, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_e=torch.tensor(0.0, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_n=torch.tensor(0.0, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_u=torch.tensor(0.0, device=device)

    scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].increment=torch.tensor(154166.666, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].initial_stroke_length=torch.tensor(0.075, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].offset=torch.tensor(0.34061, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].pivot_radius=torch.tensor(0.3204, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].initial_angle=torch.tensor(-1.570796, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].increment=torch.tensor(154166.666, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].initial_stroke_length=torch.tensor(0.075, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].offset=torch.tensor(0.3479, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].pivot_radius=torch.tensor(0.309, device=device)
    scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].initial_angle=torch.tensor(0.959931, device=device)

    incident_ray_direction = torch.tensor([0.0, -1.0, 0.0, 0.0])

    # Align heliostat.
    scenario.heliostats.heliostat_list[
        0
    ].set_aligned_surface_with_incident_ray_direction(
        incident_ray_direction=incident_ray_direction.to(device), device=device
    )

    # Create raytracer - currently only possible for one heliostat.
    raytracer = HeliostatRayTracer(
        scenario=scenario, batch_size=10
    )

    # Perform heliostat-based raytracing.
    final_bitmap = raytracer.trace_rays(
        incident_ray_direction=incident_ray_direction.to(device), device=device
    )

    final_bitmap = raytracer.normalize_bitmap(final_bitmap)

    assert final_bitmap.sum() != 0