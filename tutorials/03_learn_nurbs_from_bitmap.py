import pathlib, torch

from artist.field.surface import Surface
from artist.util.surface_converter import SurfaceConverter, NurbsConfig, FitConfig, AnalyticalConfig
from artist.util.configuration_classes import SurfaceConfig
from artist.util.load_point_clouds import load_measured_heliostat_pointcloud_from_paint_deflectometry_file, generate_ideal_juelich_heliostat_pointcloud_from_paint_heliostat_properties
from artist.util import config_dictionary, set_logger_config 
from artist.util.surface_flattener import SurfaceFlattener
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pickle

device =torch.device('cuda:2' if torch.cuda.is_available() else 'cpu')
heliostat_fp     = pathlib.Path("tutorials/data/paint/AA39/heliostat-properties.json")
deflectometry_fp = pathlib.Path("tutorials/data/paint/AA39/deflectometry.h5")   # measured cloud
ideal_fp         = None
# Define paths to save the computed surface configurations
nurbs_measured_fp = pathlib.Path("cached/nurbs_facets_measured.pkl")
nurbs_ideal_fp    = pathlib.Path("cached/nurbs_facets_ideal.pkl")

set_logger_config()
# ------------------------------------------------------------------
# 1.  generate surface from deflectometry measurements
# ------------------------------------------------------------------

(
    measured_surface_points_with_facets_list,
    measured_surface_normals_with_facets_list,
    measured_surface_translation_vectors,
    measured_surface_canting_e_vectors,
    measured_surface_canting_n_vectors,
) = load_measured_heliostat_pointcloud_from_paint_deflectometry_file(
    heliostat_file_path=heliostat_fp,
    deflectometry_file_path=deflectometry_fp,
    device=device
)

# Define NURBS and fitting configurations
nurbs_cfg = NurbsConfig(step_size=1)
fit_cfg = FitConfig(
    conversion_method=config_dictionary.convert_nurbs_from_normals,
    tolerance=3e-5,
    initial_learning_rate=1e-3,
    max_epoch=2000,
    step_size=1
)

# Create SurfaceConverter
surface_converter_measured = SurfaceConverter.from_fitting(nurbs_cfg, fit_cfg)

# Generate NURBS facet configs from point cloud, or load cached version
if nurbs_measured_fp.exists():
    with open(nurbs_measured_fp, 'rb') as f:
        nurbs_facets_measured = pickle.load(f)
else:
    nurbs_facets_measured = surface_converter_measured.fit_surface_config_from_paint(
        heliostat_file_path=heliostat_fp,
        deflectometry_file_path=deflectometry_fp,
        device=device,
    )
    with open(nurbs_measured_fp, 'wb') as f:
        pickle.dump(nurbs_facets_measured, f)

# ------------------------------------------------------------------
# 2.  Generate ideal surface from heliostat properties
# ------------------------------------------------------------------

# Generate ideal surface points and normals from the heliostat JSON
(
    ideal_surface_points_with_facets_list,
    ideal_surface_normals_with_facets_list,
    analytical_cfg
) = generate_ideal_juelich_heliostat_pointcloud_from_paint_heliostat_properties(
    heliostat_file_path=str(heliostat_fp),
    number_of_surface_points=1000,
    device=device
)

# Create converter with analytical config
surface_converter_ideal = SurfaceConverter.from_analytic(
    nurbs_cfg=nurbs_cfg,  # reuse same NurbsConfig as for measured
    analytical_cfg=analytical_cfg
)

# Load or create ideal NURBS surface
if nurbs_ideal_fp.exists():
    with open(nurbs_ideal_fp, 'rb') as f:
        nurbs_facets_ideal = pickle.load(f)
else:
    nurbs_facets_ideal = surface_converter_ideal.generate_nurbs_surface_analyticaly()
    with open(nurbs_ideal_fp, 'wb') as f:
        pickle.dump(nurbs_facets_ideal, f)




# ------------------------------------------------------------------
# 4.  Generate surface
# ------------------------------------------------------------------
flattener = SurfaceFlattener(analytical_cfg)
# Wrap facet configs in SurfaceConfig and initialize Surface


nurbs_surface_measured = Surface(SurfaceConfig(nurbs_facets_measured))
nurbs_surface_ideal = Surface(SurfaceConfig(nurbs_facets_ideal))

# Define a plotting functio

def plot_surfaces(surface_measured, surface_ideal, device):
    # 1) grab points [n_facets, n_eval, 4]
    pts_meas, _ = surface_measured.get_surface_points_and_normals(device=device)
    pts_ideal, _ = surface_ideal.get_surface_points_and_normals(device=device)

    # 2) flatten facets into one long vector and drop the 4th (homogeneous) coord
    #    result: shape [1, n_facets * n_eval, 3]
    original_surface_points = pts_meas.reshape(1, -1, 4)[..., :3]
    aligned_surface_points  = pts_ideal.reshape(1, -1, 4)[..., :3]

    colors = ["r", "g", "b", "y"]

    fig = plt.figure(figsize=(14, 6))
    gs  = fig.add_gridspec(1, 2, width_ratios=[1, 1], wspace=0.3)

    ax1 = fig.add_subplot(gs[0], projection="3d")
    ax2 = fig.add_subplot(gs[1], projection="3d")

    number_of_facets = pts_meas.shape[0]
    total_pts       = original_surface_points.shape[1]
    batch_size      = total_pts // number_of_facets

    for i in range(number_of_facets):
        start = i * batch_size
        end   = start + batch_size

        e0 = original_surface_points[0, start:end, 0].detach().cpu().numpy()
        n0 = original_surface_points[0, start:end, 1].detach().cpu().numpy()
        u0 = original_surface_points[0, start:end, 2].detach().cpu().numpy()

        e1 = aligned_surface_points[0, start:end, 0].detach().cpu().numpy()
        n1 = aligned_surface_points[0, start:end, 1].detach().cpu().numpy()
        u1 = aligned_surface_points[0, start:end, 2].detach().cpu().numpy()

        ax1.scatter(e0, n0, u0, color=colors[i], label=f"Facet {i+1}")
        ax2.scatter(e1, n1, u1, color=colors[i], label=f"Facet {i+1}")

    # labels, limits, titles
    for ax in (ax1, ax2):
        ax.set_xlabel("E"); ax.set_ylabel("N"); ax.set_zlabel("U")
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])

    # ax1.set_title("Original surface")
    # ax1.set_zlim(-0.5, 0.5)

    # ax2.set_title("Aligned surface")
    # ax2.set_ylim(4.5, 5.5)

    # one legend for both
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncols=number_of_facets)

    plt.tight_layout()
    plt.show()
    fig.savefig("tut_3.png")

# Call the plotting function
plot_surfaces(nurbs_surface_measured, nurbs_surface_ideal, device)