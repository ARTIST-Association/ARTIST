import logging
import struct
import sys
from typing import List, Optional, Tuple, cast

import colorlog
import numpy as np
import torch

from artist.util.nurbs_converters import deflectometry_to_nurbs, point_cloud_to_nurbs

Tuple3d = Tuple[np.floating, np.floating, np.floating]
Vector3d = List[np.floating]


class StralConverter:
    """
    This class implements a converter that converts STRAL data to HDF5 format.

    Attributes
    ----------
    stral_file_path : str
        The file path to the STRAL data file that will be converted.
    hdf5_file_path : str
        The file path for the HDF5 file that will be saved.
     concentrator_header_name : str
        The name for the concentrator header in the STRAL file.
    facet_header_name : str
        The name for the facet header in the STRAL file.
    ray_struct_name : str
        The name of the ray structure in the STRAL file.
    step_size : int
        The size of the step used to reduce the number of considered points for compute efficiency.
    log : logging.Logger
        The logger.

    Methods
    -------
    convert_point_to_4d_format()
        Convert a 3D point to a 4D point.
    convert_direction_to_4d_format()
        Convert a 3D direction vector to 4D format.
    nwu_to_enu()
        Cast from an NWU to an ENU coordinate system.
    convert_stral_to_hdf5()
        Convert the STRAL data to HDF5 data.

    """

    def __init__(
        self,
        stral_file_path: str,
        hdf5_file_path: str,
        concentrator_header_name: str,
        facet_header_name: str,
        ray_struct_name: str,
        step_size: int,
        log_level: Optional[int] = logging.INFO,
    ) -> None:
        """
        Initialize the converter.

        Parameters
        ----------
        stral_file_path : str
            The file path to the STRAL data file that will be converted.
        hdf5_file_path : str
            The file path for the HDF5 file that will be saved.
        concentrator_header_name : str
            The name for the concentrator header in the STRAL file.
        facet_header_name : str
            The name for the facet header in the STRAL file.
        ray_struct_name : str
            The name of the ray structure in the STRAL file.
        step_size : int
            The size of the step used to reduce the number of considered points for compute efficiency.
        log_level : Optional[int]
            The log level used for the logger.
        """
        self.stral_file_path = stral_file_path
        self.hdf5_file_path = hdf5_file_path
        self.concentrator_header_name = concentrator_header_name
        self.facet_header_name = facet_header_name
        self.ray_struct_name = ray_struct_name
        self.step_size = step_size
        log = logging.getLogger("STRAL-to-h5-converter")  # Get logger instance.
        log_formatter = colorlog.ColoredFormatter(
            fmt="[%(cyan)s%(asctime)s%(reset)s][%(blue)s%(name)s%(reset)s]"
            "[%(log_color)s%(levelname)s%(reset)s] - %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "red,bg_white",
            },
            secondary_log_colors={},
        )
        handler = logging.StreamHandler(stream=sys.stdout)
        handler.setFormatter(log_formatter)
        log.addHandler(handler)
        log.setLevel(log_level)
        self.log = log

    @staticmethod
    def convert_3d_points_to_4d_format(point: torch.Tensor) -> torch.Tensor:
        """
        Append ones to the last dimension of a PyTorch tensor.

        Includes convention that points have a 1 and directions have 0 as 4th dimension

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            Tensor with ones appended along the last dimension.

        Examples
        --------
        >>> import torch
        >>> tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> new_tensor = convert_3d_points_to_4d_format(tensor)
        >>> print(new_tensor)
        tensor([[[1., 2., 1.],
                [3., 4., 1.]],

                [[5., 6., 1.],
                [7., 8., 1.]]])
        """
        ones_shape = list(point.shape)
        ones_shape[-1] = 1
        ones_tensor = torch.ones(ones_shape, dtype=point.dtype, device=point.device)
        return torch.cat((point, ones_tensor), dim=-1)

    @staticmethod
    def convert_direction_to_4d_format(direction: torch.Tensor) -> torch.Tensor:
        """
        Append ones to the last dimension of a PyTorch tensor.

        Includes convention that points have a 1 and directions have 0 as 4th dimension.

        Parameters
        ----------
        tensor : torch.Tensor
            Input tensor of any shape.

        Returns
        -------
        torch.Tensor
            Tensor with ones appended along the last dimension.

        Examples
        --------
        >>> import torch
        >>> tensor = torch.tensor([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])
        >>> new_tensor = convert_direction_to_4d_format(tensor)
        >>> print(new_tensor)
        tensor([[[1., 2., 0.],
                [3., 4., 0.]],

                [[5., 6., 0.],
                [7., 8., 0.]]])
        """
        zeros_shape = list(direction.shape)
        zeros_shape[-1] = 1
        zeros_tensor = torch.zeros(
            zeros_shape, dtype=direction.dtype, device=direction.device
        )
        return torch.cat((direction, zeros_tensor), dim=-1)

    @staticmethod
    def nwu_to_enu(vec: Tuple3d) -> Vector3d:
        """
        Cast the coordinate system from NWU to ENU.

        Parameters
        ----------
        vec : Tuple3d
            The vector that is to be casted.

        Returns
        -------
        Vector3d
            The casted vector in the ENU coordinate system.
        """
        return [-vec[1], vec[0], vec[2]]

    def convert_stral_file_to_nurbs(
        self,
        extraction_method: str,  # Not used at the moment, placeholder for future.
        num_control_points_e: int,
        num_control_points_n: int,
        tolerance: float = 1.2e-6,
        max_epoch: int = 10000,
    ) -> List[torch.Tensor]:
        """
        Extract information from a STRAL file saved as .binp and save this information as an HDF5 file.

        Parameters
        ----------
        extraction_method : str
            Placeholder for future extraction methods.
        num_control_points_e : int
            Number of control points in the east direction.
        num_control_points_n : int
            Number of control points in the north direction.
        tolerance : float
            Tolerance to use when fitting NURBS surfaces.
        max_epoch : int
            Maximum number of epochs to use when fitting NURBS surfaces.
        """
        self.log.info("Beginning STRAL to HDF5 conversion!")

        # Create structures for reading STRAL file correctly.
        concentrator_header_struct = struct.Struct(self.concentrator_header_name)
        facet_header_struct = struct.Struct(self.facet_header_name)
        ray_struct = struct.Struct(self.ray_struct_name)

        with open(f"{self.stral_file_path}.binp", "rb") as file:
            byte_data = file.read(concentrator_header_struct.size)
            concentrator_header_data = concentrator_header_struct.unpack_from(byte_data)
            self.log.info(f"Reading STRAL file located at: {self.stral_file_path}")

            # Load surface position.
            surface_position = self.nwu_to_enu(
                cast(Tuple3d, concentrator_header_data[0:3])
            )

            # Load width and height.
            width, height = concentrator_header_data[3:5]

            # Calculate number of facets.
            n_xy = concentrator_header_data[5:7]
            number_of_facets = n_xy[0] * n_xy[1]

            # Create empty lists for storing data.
            facet_positions: List[Vector3d] = []
            facet_spans_n: List[Vector3d] = []
            facet_spans_e: List[Vector3d] = []
            surface_points_faceted: List[List[Vector3d]] = [
                [] for _ in range(number_of_facets)
            ]
            surface_normals_faceted: List[List[Vector3d]] = [
                [] for _ in range(number_of_facets)
            ]
            surface_ideal_vectors: List[List[Vector3d]] = [
                [] for _ in range(number_of_facets)
            ]

            normals = []

            for f in range(number_of_facets):
                byte_data = file.read(facet_header_struct.size)
                facet_header_data = facet_header_struct.unpack_from(byte_data)

                facet_pos = cast(Tuple3d, facet_header_data[1:4])
                facet_vec_x = np.array(
                    [
                        -facet_header_data[5],
                        facet_header_data[4],
                        facet_header_data[6],
                    ]
                )
                facet_vec_y = np.array(
                    [
                        -facet_header_data[8],
                        facet_header_data[7],
                        facet_header_data[9],
                    ]
                )

                facet_vec_z = np.cross(facet_vec_x, facet_vec_y)

                facet_positions.append(facet_pos)

                facet_spans_n.append(facet_vec_x.tolist())
                facet_spans_e.append(facet_vec_y.tolist())

                ideal_normal = (facet_vec_z / np.linalg.norm(facet_vec_z)).tolist()
                normals.append(ideal_normal)
                n_rays = facet_header_data[10]

                byte_data = file.read(ray_struct.size * n_rays)
                ray_datas = ray_struct.iter_unpack(byte_data)

                for ray_data in ray_datas:
                    surface_points_faceted[f].append(cast(Tuple3d, ray_data[:3]))
                    surface_normals_faceted[f].append(cast(Tuple3d, ray_data[3:6]))
                    surface_ideal_vectors[f].append(ideal_normal)

        self.log.info("Loading STRAL data complete")

        # Stral uses two different coordinate systems, both use a west orientation and therefore, we don't need an NWU
        # to ENU cast here. However, to maintain consistency we cast the west direction to east direction.
        for span_e in facet_spans_e:
            span_e[0] = -span_e[0]

        # Select only selected number of points to reduce compute.
        surface_points_faceted = torch.tensor(surface_points_faceted)[
            :, :: self.step_size
        ]
        surface_normals_faceted = torch.tensor(surface_normals_faceted)[
            :, :: self.step_size
        ]
        surface_ideal_vectors = torch.tensor(surface_ideal_vectors)[
            :, :: self.step_size
        ]

        # Convert to torch tensor.
        surface_position = torch.tensor(surface_position)
        facet_positions = torch.tensor(facet_positions)
        facet_spans_n = torch.tensor(facet_spans_n)
        facet_spans_e = torch.tensor(facet_spans_e)

        # Convert to 4D format.
        surface_position = self.convert_3d_points_to_4d_format(surface_position)
        facet_positions = self.convert_3d_points_to_4d_format(facet_positions)
        facet_spans_n = self.convert_direction_to_4d_format(facet_spans_n)
        facet_spans_e = self.convert_direction_to_4d_format(facet_spans_e)
        surface_points_faceted = self.convert_3d_points_to_4d_format(
            surface_points_faceted
        )
        surface_normals_faceted = self.convert_direction_to_4d_format(
            surface_normals_faceted
        )
        surface_ideal_vectors = self.convert_direction_to_4d_format(
            surface_ideal_vectors
        )

        # Convert to NURBS surface.
        self.log.info("Converting to NURBS surface")
        facetted_nurbs = []
        if extraction_method == "point_cloud":
            self.log.info("Fit NURBS to point cloud")
            for i, points_on_facet in enumerate(surface_points_faceted):
                self.log.info(f"Optimize {i+1} of {number_of_facets} facets")
                facetted_nurbs.append(
                    point_cloud_to_nurbs(
                        points_on_facet,
                        num_control_points_e,
                        num_control_points_n,
                        tolerance,
                        max_epoch,
                    )
                )
        elif extraction_method == "deflectometry":
            self.log.info("Fit NURBS to deflectometry measurement")
            for i, (points_on_facet, normals_on_facet) in enumerate(
                zip(surface_points_faceted, surface_normals_faceted)
            ):
                self.log.info(f"Optimize {i+1} of {number_of_facets} facets")
                facetted_nurbs.append(
                    deflectometry_to_nurbs(
                        points_on_facet,
                        normals_on_facet,
                        num_control_points_e,
                        num_control_points_n,
                        tolerance,
                        max_epoch,
                    )
                )
        return facetted_nurbs

        # self.log.info(f"Creating HDF5 file in location: {self.hdf5_file_path}")

        # with h5py.File(f"{self.hdf5_file_path}.h5", "w") as f:
