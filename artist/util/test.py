import pathlib
import torch
import pathlib
import torch
import matplotlib.pyplot as plt

from artist.field.facets_nurbs import NurbsFacet
from artist.util.surface_converter import SurfaceConverter
from artist.util.configuration_classes import FacetConfig

from artist.util import config_dictionary

def get_or_create_facet_list(surface_converter,
                             deflectometry_file_path: pathlib.Path,
                             heliostat_file_path: pathlib.Path,
                             device: torch.device,
                             cache_file_path: pathlib.Path) -> list[FacetConfig]:
    """
    Retrieves the facet list from a cache file if it exists. Otherwise, generates the facet list,
    saves it to the cache file, and returns the generated data.

    Parameters
    ----------
    surface_converter : SurfaceConverter
        An instance of SurfaceConverter used to generate the facet list.
    deflectometry_file_path : pathlib.Path
        Path to the deflectometry data file.
    heliostat_file_path : pathlib.Path
        Path to the heliostat properties file.
    device : torch.device
        The device on which the computation runs.
    cache_file_path : pathlib.Path
        File path where the facet list is cached.

    Returns
    -------
    any
        The facet list generated from the input files.
    """
    if cache_file_path.exists():
        with torch.serialization.safe_globals([FacetConfig]):
            facet_list = torch.load(cache_file_path)
        print(f"Loading cached facet list from {cache_file_path}")
    else:
        print("Cached facet list not found. Generating facet list...")
        facet_list = surface_converter.generate_surface_config_from_paint(
            deflectometry_file_path=deflectometry_file_path,
            heliostat_file_path=heliostat_file_path,
            device=device,
        )
        torch.save(facet_list, cache_file_path)
        print(f"Facet list saved to {cache_file_path}")
    return facet_list


# Import your get_or_create_facet_list utility (e.g., from your test.py file)

# Set device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def save_surface_plot(surface_points: torch.Tensor, title: str, filename: str):
    """
    Plot and save a set of 3D points as a scatter plot.
    
    Parameters
    ----------
    surface_points : torch.Tensor
        Tensor of shape (N, 4); only the first three dimensions are plotted.
    title : str
        Title for the plot.
    filename : str
        File path where the plot will be saved.
    """
    pts = surface_points[:, :3].detach().cpu().numpy()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], s=1)
    ax.set_title(title)
    plt.savefig(filename)
    plt.close(fig)


if __name__ == "__main__":
    # Define file paths (update these paths as needed)
    deflectometry_fp = pathlib.Path("/workVERLEIHNIX/mp/ARTIST/tests/data/field_data/AA39-deflectometry.h5")
    heliostat_fp = pathlib.Path("/workVERLEIHNIX/mp/ARTIST/tests/data/field_data/AA39-heliostat-properties.json")
    cache_fp = pathlib.Path("cached_facet_list.pt")  # cache file in current directory
    # Create a SurfaceConverter

    surface_converter = SurfaceConverter(step_size=1, max_epoch=20000, conversion_method=config_dictionary.convert_nurbs_from_points)

    # Get the facet list (either from cache or by generating it)
    facet_list = get_or_create_facet_list(
        surface_converter=surface_converter,
        deflectometry_file_path=None,#deflectometry_fp,
        heliostat_file_path=heliostat_fp,
        device=device,
        cache_file_path=cache_fp
    )
    print("Facet list loaded:", facet_list)
    
all_surface_points = []
for facet in facet_list:
    control_points = facet.control_points.to(device)
    translation_vector = facet.translation_vector.to(device)
    # Create a NurbsFacet instance from the configuration
    nurbs_facet = NurbsFacet(
        control_points=control_points,
        degree_e=facet.degree_e,
        degree_n=facet.degree_n,
        number_eval_points_e=facet.number_eval_points_e,
        number_eval_points_n=facet.number_eval_points_n,
        translation_vector=translation_vector,

        canting_e=facet.canting_n,
        canting_n=facet.canting_e,
    )
    # Use the NurbsFacet to create a NURBS surface
    nurbs_surface = nurbs_facet.create_nurbs_surface(device=device)
    pts, _ = nurbs_surface.calculate_surface_points_and_normals(device=device)
    pts = pts + translation_vector  # adjust by facet translation
    all_surface_points.append(pts)
deflectometry_surface = torch.cat(all_surface_points, dim=0)
# Save deflectometry surface plot.
save_surface_plot(deflectometry_surface, title="Deflectometry Surface", filename="deflectometry_surface3.png")
