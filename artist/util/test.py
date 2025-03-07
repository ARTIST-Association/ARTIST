# Generate surface configuration from data.
import pathlib
import torch
from artist.util.surface_converter import SurfaceConverter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

surface_converter = SurfaceConverter(
    step_size=100,
    max_epoch=400,
)

facet_list = surface_converter.generate_surface_config_from_paint(
    deflectometry_file_path=pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tests/data/field_data/AA39-deflectometry.h5"),
    heliostat_file_path=pathlib.Path("/workVERLEIHNIX/mb/ARTIST/tests/data/field_data/AA39-heliostat-properties.json"),
    device=device,
)