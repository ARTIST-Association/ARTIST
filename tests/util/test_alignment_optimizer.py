import json
import pathlib

from artist.raytracing.heliostat_tracing import HeliostatRayTracer
from artist.scenario import Scenario
from artist.util import utils
import h5py
import pytest
import torch

from artist import ARTIST_ROOT
from artist.util.alignment_optimizer import AlignmentOptimizer

@pytest.fixture(params=["cpu", "cuda:3"] if torch.cuda.is_available() else ["cpu"])
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

@pytest.mark.parametrize("optimizer_method, expected", [
    ("optimize_kinematic_parameters_with_motor_positions", 0),
    ("optimize_kinematic_parameters_with_raytracing", 0)
])
def test_alignment_optimizer(optimizer_method, device, expected):
    """
    TODO docstrings
    """
    scenario_path = pathlib.Path(ARTIST_ROOT) / "scenarios/test_scenario_paint.h5"
    calibration_properties_path = (
        pathlib.Path(ARTIST_ROOT)
        / "measurement_data/download_test/AA39/Calibration/86500-calibration-properties.json"
    )

    alignment_optimizer = AlignmentOptimizer(
        scenario_path=scenario_path,
        calibration_properties_path=calibration_properties_path,
    )

    optimized_kinematic_parameters = getattr(alignment_optimizer, optimizer_method)(device=device)

    #assert True
    # # Assertion

    # # Load the scenario.
    # with h5py.File(scenario_path, "r") as config_h5:
    #     scenario = Scenario.load_scenario_from_hdf5(
    #         scenario_file=config_h5, device=device
    #     )

    # # Load the calibration data.
    # calibration_item_stac_file = pathlib.Path(ARTIST_ROOT) /"measurement_data/download_test/AA39/Calibration/86500-calibration-item-stac.json"
    # with open(calibration_item_stac_file, 'r') as file:
    #     calibration_dict = json.load(file)
    #     sun_azimuth = torch.tensor(calibration_dict["view:sun_azimuth"], device=device)
    #     sun_elevation = torch.tensor(calibration_dict["view:sun_elevation"], device=device)
    #     incident_ray_direction = utils.convert_3d_direction_to_4d_format(utils.azimuth_elevation_to_enu(sun_azimuth, sun_elevation, degree=True), device=device)

    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_e = optimized_kinematic_parameters[0]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_n = optimized_kinematic_parameters[1]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_translation_u = optimized_kinematic_parameters[2]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_e = optimized_kinematic_parameters[3]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_n = optimized_kinematic_parameters[4]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.first_joint_tilt_u = optimized_kinematic_parameters[5]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_e = optimized_kinematic_parameters[6]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_n = optimized_kinematic_parameters[7]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_translation_u = optimized_kinematic_parameters[8]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_e = optimized_kinematic_parameters[9]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_n = optimized_kinematic_parameters[10]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.second_joint_tilt_u = optimized_kinematic_parameters[11]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_e = optimized_kinematic_parameters[12]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_n = optimized_kinematic_parameters[13]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_translation_u = optimized_kinematic_parameters[14]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_e = optimized_kinematic_parameters[15]                 
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_n = optimized_kinematic_parameters[16]
    # scenario.heliostats.heliostat_list[0].kinematic.deviation_parameters.concentrator_tilt_u = optimized_kinematic_parameters[17]
    # scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].increment = optimized_kinematic_parameters[18]
    # scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].initial_stroke_length = optimized_kinematic_parameters[19]
    # scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].offset = optimized_kinematic_parameters[20]
    # scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].radius = optimized_kinematic_parameters[21]
    # scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[0].phi_0 = optimized_kinematic_parameters[22]
    # scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].increment = optimized_kinematic_parameters[23]
    # scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].initial_stroke_length = optimized_kinematic_parameters[24]
    # scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].offset = optimized_kinematic_parameters[25]
    # scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].radius = optimized_kinematic_parameters[26]
    # scenario.heliostats.heliostat_list[0].kinematic.actuators.actuator_list[1].phi_0 = optimized_kinematic_parameters[27]

    # # Align heliostat.
    # scenario.heliostats.heliostat_list[0].set_aligned_surface(
    #     incident_ray_direction=incident_ray_direction.to(device), device=device
    # )

    # # Create raytracer - currently only possible for one heliostat.
    # raytracer = HeliostatRayTracer(
    #     scenario=scenario
    # )

    # # Perform heliostat-based raytracing.
    # final_bitmap = raytracer.trace_rays(
    #     incident_ray_direction=incident_ray_direction.to(device), device=device
    # )

    # final_bitmap = raytracer.normalize_bitmap(final_bitmap)
    # torch.save(final_bitmap, "bitmap_test.pt")

    # # import matplotlib.pyplot as plt
    # # plt.imshow(final_bitmap.cpu().detach().numpy())
    # # plt.savefig("test1.png")

    # # assert True
