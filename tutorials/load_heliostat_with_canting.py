import pathlib, torch

from artist.field.surface import Surface
from artist.util.surface_converter import SurfaceConverter
from artist.util.configuration_classes import SurfaceConfig
from artist.util.generate_point_clouds import generate_ideal_juelich_heliostat_pointcloud_from_paint_heliostat_properties, load_measured_heliostat_pointcloud_from_paint_deflectometry_file
from artist.util import config_dictionary, set_logger_config 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

device = torch.device('cuda:3' if torch.cuda.is_available() else 'cpu')
heliostat_fp     = pathlib.Path("tutorials/data/test_scenario_surface_optimization_AA39/AA39/Properties/AA39-heliostat-properties.json")
deflectometry_fp = pathlib.Path("tutorials/data/test_scenario_surface_optimization_AA39/AA39/Deflectometry/AA39-filled-2023-09-18Z08-49-09Z-deflectometry.h5")   # measured cloud
ideal_fp         = None
# Define paths to save the computed surface configurations
nurbs_measured_fp = pathlib.Path("cached/nurbs_facets_measured.pkl")
nurbs_ideal_fp    = pathlib.Path("cached/nurbs_facets_ideal.pkl")

set_logger_config()
# ------------------------------------------------------------------
# 1.  Load and construct point clouds from measured surfaces and ideal properties
# ------------------------------------------------------------------
number_of_surface_points_for_meausred_surface = 2000 # None = all points 

(
    measured_surface_points_with_facets_list, 
    measured_surface_normals_with_facets_list,
    measured_surface_translation_vectors
    ) = load_measured_heliostat_pointcloud_from_paint_deflectometry_file(heliostat_file_path = heliostat_fp, 
                                                                        deflectometry_file_path = deflectometry_fp, 
                                                                        device = device)


number_of_surface_points_for_ideal_surface = 2000
(
    ideal_surface_points_with_facets_list, 
    ideal_surface_normals_with_facets_list, 
    ideal_surface_translation_vectors
     ) = generate_ideal_juelich_heliostat_pointcloud_from_paint_heliostat_properties(
        heliostat_file_path = heliostat_fp,
        number_of_surface_points = 1000,
        device = device,
    )


# ------------------------------------------------------------------
# 2.  Converter settings – in general use POINTS for the ideal case and normals for the measured case
# ------------------------------------------------------------------
surface_converter_measured = SurfaceConverter(
        conversion_method = config_dictionary.convert_nurbs_from_normals,
        step_size         = 1,
        max_epoch         = 2000,
)

surface_converter_ideal = SurfaceConverter(
        conversion_method = config_dictionary.convert_nurbs_from_points,
        step_size         = 1,
        max_epoch         = 2000,
)

# ------------------------------------------------------------------
# 3.  Generate surface config from point clouds
# ------------------------------------------------------------------

# Create the directory if it doesn't exist
nurbs_measured_fp.parent.mkdir(parents=True, exist_ok=True)

if nurbs_measured_fp.exists():
    with open(nurbs_measured_fp, 'rb') as f:
        nurbs_facets_measured = pickle.load(f)
else:
    nurbs_facets_measured = surface_converter_measured.generate_nurbs_surface_config_from_pointcloud(
        surface_points_with_facets_list = measured_surface_points_with_facets_list,
        surface_normals_with_facets_list = measured_surface_normals_with_facets_list,
        facet_translation_vectors = measured_surface_translation_vectors,
        device = device,
    )
    with open(nurbs_measured_fp, 'wb') as f:
        pickle.dump(nurbs_facets_measured, f)

# Load or generate ideal NURBS surface config
if nurbs_ideal_fp.exists():
    with open(nurbs_ideal_fp, 'rb') as f:
        nurbs_facets_ideal = pickle.load(f)
else:
    nurbs_facets_ideal = surface_converter_ideal.generate_nurbs_surface_config_from_pointcloud(
        surface_points_with_facets_list = ideal_surface_points_with_facets_list,
        surface_normals_with_facets_list = ideal_surface_normals_with_facets_list,
        facet_translation_vectors = ideal_surface_translation_vectors,
        device = device,
    )
    with open(nurbs_ideal_fp, 'wb') as f:
        pickle.dump(nurbs_facets_ideal, f)

nurbs_surface_config_ideal = SurfaceConfig(facet_list=nurbs_facets_ideal)
nurbs_surface_config_meausred = SurfaceConfig(facet_list=nurbs_facets_measured)

# ------------------------------------------------------------------
# 4.  Generate surface
# ------------------------------------------------------------------

nurbs_surface = Surface(surface_config_measured = nurbs_facets_measured, surface_config_ideal = nurbs_facets_ideal)

nurbs_surface.get_surface_points_and_normals()

def plot_control_point_surfaces(nurbs_surface):
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')

    for i, facet in enumerate(nurbs_surface.facets):
        # Each of these is a list of 4 tensors, one per facet, each of shape (M, N, 3)
        print(facet.control_points_ideal[i])
        print(facet.control_points_measured[i])
        ideal_cp = facet.control_points_ideal[i]      # Ideal control points
        measured_cp = facet.control_points_measured[i]  # Deviation (measured - ideal)
        combined_cp = measured_cp + ideal_cp          # Actual control points

        for cp, label, color in zip(
            [ideal_cp, measured_cp + ideal_cp, combined_cp],
            ['Ideal', 'Measured', 'Combined'],
            ['blue', 'red', 'green']
        ):
            X = cp[:, :, 0].cpu().numpy()
            Y = cp[:, :, 1].cpu().numpy()
            Z = cp[:, :, 2].cpu().numpy()

            ax.plot_surface(X, Y, Z, alpha=0.5, color=color, edgecolor='k', linewidth=0.5)

    ax.set_title("NURBS Control Points: Ideal, Measured, Combined")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    legend_labels = ['Ideal', 'Measured', 'Combined']
    proxy = [plt.Rectangle((0, 0), 1, 1, fc=color, alpha=0.5) for color in ['blue', 'red', 'green']]
    ax.legend(proxy, legend_labels)

    plt.tight_layout()
    plt.savefig("testsurface.png")

plot_control_point_surfaces(nurbs_surface)

