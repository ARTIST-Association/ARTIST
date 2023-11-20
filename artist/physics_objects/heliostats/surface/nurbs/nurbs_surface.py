
from typing import List, Optional, Tuple, Union
from yacs.config import CfgNode
import torch

from .....util import utils
from ..nurbs import nurbs


class ANURBSSurface():
    def calc_normals_and_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        raise NotImplementedError(
            'please override `_calc_normals_and_surface`')

    def discrete_points_and_normals(self) -> Tuple[torch.Tensor, torch.Tensor]:
        discrete_points, normals = self.calc_normals_and_surface()
        return discrete_points, normals

    @property
    def discrete_points(self) -> torch.Tensor:
        discrete_points, _ = self.calc_normals_and_surface()
        return discrete_points

    @property
    def normals(self) -> torch.Tensor:
        _, normals = self.calc_normals_and_surface()
        return normals
    
class NURBSSurface(ANURBSSurface):
    def __init__(
            self,
            nurbs_config: CfgNode,
            device: torch.device,
            position_on_field: torch.Tensor,
    ) -> None:
        super().__init__()
        self.nurbs_cfg = nurbs_config
        self.device = device
        self.position_on_field = position_on_field
        if not self.nurbs_cfg.is_frozen():
            self.nurbs_cfg = self.nurbs_cfg.clone()
            self.nurbs_cfg.freeze()

        self._fix_spline_ctrl_weights: bool = \
            self.nurbs_cfg.FIX_SPLINE_CTRL_WEIGHTS
        self._fix_spline_knots: bool = self.nurbs_cfg.FIX_SPLINE_KNOTS
        self._recalc_eval_points: bool = self.nurbs_cfg.RECALCULATE_EVAL_POINTS

        spline_degree: int = self.nurbs_cfg.SPLINE_DEGREE
        self.degree_x = spline_degree
        self.degree_y = spline_degree
        self.h_rows: Optional[int] = None
        self.h_cols: Optional[int] = None
        self.rows: int = self.nurbs_cfg.ROWS
        self.cols: int = self.nurbs_cfg.COLS

        self._progressive_growing = ProgressiveGrowing(self)

        (
            ctrl_points,
            ctrl_weights,
            knots_x,
            knots_y,
        ) = nurbs.setup_nurbs_surface(
            self.degree_x, self.degree_y, self.rows, self.cols, self.device)

        utils.initialize_spline_knots(
            knots_x, knots_y, self.degree_x, self.degree_y)
        ctrl_weights[:] = 1
  
        self.initialize_control_points(ctrl_points)
        self.split_nurbs_params(ctrl_points, ctrl_weights, knots_x, knots_y)
        self.initialize_eval_points()

    
    def initialize_control_points(self, ctrl_points: torch.Tensor) -> None:
        """
        Initialize the control points.

        Parameters
        ----------
        ctrl_points : torch.Tensor
            The control points of the nurbs surface.
        """
        nurbs_config = self.nurbs_cfg

        width = nurbs_config.WIDTH
        height = nurbs_config.HEIGHT

        print(torch.zeros_like(self.position_on_field))

        utils.initialize_spline_ctrl_points(
            ctrl_points,
            # We are only moved to `position_on_field` upon alignment,
            # so initialize at the origin where the heliostat's discrete
            # points are as well.
            torch.zeros_like(self.position_on_field),
            self.rows,
            self.cols,
            width,
            height,
        )

    def split_nurbs_params(
            self,
            ctrl_points: torch.Tensor,
            ctrl_weights: torch.Tensor,
            knots_x: torch.Tensor,
            knots_y: torch.Tensor,
    ) -> None:
        """
        Split the nurbs parameters along their dimensions.

        Paraneters
        ----------
        ctrl_points : torch.Tensor
            The control points of the nurbs.
        ctrl_weights : torch.Tensor,
            The weights of the control points.
        knots_x : torch.Tensor
            List of numbers representing the knots in x dimension.
        knots_y : torch.Tensor
            List of numbers representing the knots in y dimension. 
        """
        with torch.no_grad():
            self.set_ctrl_points(ctrl_points)
            self.set_ctrl_weights(ctrl_weights)
            self.set_knots_x(knots_x)
            self.set_knots_y(knots_y)
    
    def set_ctrl_points(self, ctrl_points: torch.Tensor) -> None:
        """
        Split the control points in the last dimension then column dimension, then row dimension.

        Parameters
        ----------
        ctrl_points : torch.Tensor
            The control points of the nurbs.
        """
        with torch.no_grad():
            first_row = [
                [row_slice[:, :, dim].unsqueeze(-1) for dim in range(3)]
                for row_slice in [
                        ctrl_points[:1, :1],
                        ctrl_points[:1, 1:-1],
                        ctrl_points[:1, -1:],
                ]
            ]
            inner_rows = [
                [rows_slice[:, :, dim].unsqueeze(-1) for dim in range(3)]
                for rows_slice in [
                        ctrl_points[1:-1, :1],
                        ctrl_points[1:-1, 1:-1],
                        ctrl_points[1:-1, -1:],
                ]
            ]
            last_row = [
                [row_slice[:, :, dim].unsqueeze(-1) for dim in range(3)]
                for row_slice in [
                        ctrl_points[-1:, :1],
                        ctrl_points[-1:, 1:-1],
                        ctrl_points[-1:, -1:],
                ]
            ]

            self._ctrl_points_splits = [first_row, inner_rows, last_row]
            assert (self.ctrl_points == ctrl_points).all()

    def set_ctrl_weights(self, ctrl_weights: torch.Tensor) -> None:
        """
        Split the control weights in column dimension, then row dimension.

        Parameters
        ----------
        ctrl_weights : torch.Tensor
            The weights of the control points of the nurbs.
        """
        with torch.no_grad():
            first_row = [
                ctrl_weights[:1, :1],
                ctrl_weights[:1, 1:-1],
                ctrl_weights[:1, -1:],
            ]
            inner_rows = [
                ctrl_weights[1:-1, :1],
                ctrl_weights[1:-1, 1:-1],
                ctrl_weights[1:-1, -1:],
            ]
            last_row = [
                ctrl_weights[-1:, :1],
                ctrl_weights[-1:, 1:-1],
                ctrl_weights[-1:, -1:],
            ]

            self._ctrl_weights_splits = [first_row, inner_rows, last_row]
            assert (self.ctrl_weights == ctrl_weights).all()

    def set_knots_x(self, knots_x: torch.Tensor) -> None:
        """
        Set the knots in x dimension.

        Parameters
        ----------
        knots_x : torch.Tensor
            The knots of the x dimension.
        """
        with torch.no_grad():
            self._knots_x_splits = self._split_knots(knots_x)
            assert (self.knots_x == knots_x).all()

    def set_knots_y(self, knots_y: torch.Tensor) -> None:
        """
        Set the knots in y dimension.

        Parameters
        ----------
        knots_y : torch.Tensor
            The knots of the y dimension.
        """
        with torch.no_grad():
            self._knots_y_splits = self._split_knots(knots_y)
            assert (self.knots_y == knots_y).all()
    
    @staticmethod
    def _split_knots(knots: torch.Tensor) -> List[torch.Tensor]:
        """
        Split the knots into three groups.

        Parameters
        ----------
        knots : torch.Tensor
            List of knots to be split.
        
        Returns 
        List[torch.Tensor]
            The splitted knots.
        """
        with torch.no_grad():
            return [knots[:1], knots[1:-1], knots[-1:]]
        
    @property
    def ctrl_points(self) -> torch.Tensor:
        # self._ctrl_points_splits[0][0] per-dim
        # self._ctrl_points_splits[0] per-column
        # self._ctrl_points_splits per-row
        first_row = torch.cat([
            torch.cat(row_slice, dim=-1)
            for row_slice in self._ctrl_points_splits[0]
        ], dim=1)
        inner_rows = torch.cat([
            torch.cat(rows_slice, dim=-1)
            for rows_slice in self._ctrl_points_splits[1]
        ], dim=1)
        last_row = torch.cat([
            torch.cat(row_slice, dim=-1)
            for row_slice in self._ctrl_points_splits[2]
        ], dim=1)

        return torch.cat([first_row, inner_rows, last_row], dim=0)
    
    @property
    def ctrl_weights(self) -> torch.Tensor:
        first_row = torch.cat(self._ctrl_weights_splits[0], dim=1)
        inner_rows = torch.cat(self._ctrl_weights_splits[1], dim=1)
        last_row = torch.cat(self._ctrl_weights_splits[2], dim=1)
        return torch.cat([first_row, inner_rows, last_row], dim=0)
    
    @property
    def knots_x(self) -> torch.Tensor:
        return torch.cat(self._knots_x_splits)
    
    @property
    def knots_y(self) -> torch.Tensor:
        return torch.cat(self._knots_y_splits)
    

    def initialize_eval_points(self) -> None:
        """
        Initialize the spline evaluation points.
        """
        # Unless we change the knots, we don't need to recalculate
        # as we simply distribute the points uniformly.
        self._recalc_eval_points = False
        self._eval_points = utils.initialize_spline_eval_points(
            self.rows, self.cols, self.device)


    def calc_normals_and_surface(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Caluclate the surface points and the surface normals of the nurbs.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            The surface points and surface normals.
        """
        eval_points = self.eval_points
        ctrl_points, ctrl_weights, knots_x, knots_y = \
            self._progressive_growing.select()

        surface_points, normals = nurbs.calc_normals_and_surface_slow(
            eval_points[:, 0],
            eval_points[:, 1],
            self.degree_x,
            self.degree_y,
            ctrl_points,
            ctrl_weights,
            knots_x,
            knots_y,
        )

        return surface_points, normals

    @property
    def eval_points(self) -> torch.Tensor:
        eval_points = self._eval_points
        return eval_points
        

class ProgressiveGrowing:
    def __init__(self, heliostat: 'NURBSSurface') -> None:
        self.heliostat = heliostat
        self.cfg: CfgNode = self.heliostat.nurbs_cfg.GROWING

        self._interval: int = self.cfg.INTERVAL
        self._step = 0

        if self._no_progressive_growing():
            self.row_indices: Optional[torch.Tensor] = None
            self.col_indices: Optional[torch.Tensor] = None
            return

        self.device = self.heliostat.device

        self.row_indices = self._calc_start_indices(
            self.cfg.START_ROWS,
            self.heliostat.rows,
            self.heliostat.degree_x,
        )
        self.col_indices = self._calc_start_indices(
            self.cfg.START_COLS,
            self.heliostat.cols,
            self.heliostat.degree_y,
        )

    def select(
            self,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        degree_x = self.heliostat.degree_x
        degree_y = self.heliostat.degree_y
        ctrl_points = self.heliostat.ctrl_points
        ctrl_weights = self.heliostat.ctrl_weights
        knots_x = self.heliostat.knots_x
        knots_y = self.heliostat.knots_y
        indices_x = self.row_indices
        indices_y = self.col_indices

        if indices_x is not None:
            ctrl_points = ctrl_points[indices_x, :]
            ctrl_weights = ctrl_weights[indices_x, :]

            # FIXME can we make this more dynamic, basing it on the
            #       available knot points?
            knots_x = torch.empty(
                len(indices_x) + degree_x + 1,
                device=self.device,
                dtype=knots_x.dtype,
            )
            utils.initialize_spline_knots_(knots_x, degree_x)

        if indices_y is not None:
            ctrl_points = ctrl_points[:, indices_y]
            ctrl_weights = ctrl_weights[:, indices_y]

            # FIXME can we make this more dynamic, basing it on the
            #       available knot points?
            knots_y = torch.empty(
                len(indices_y) + degree_y + 1,
                device=self.device,
                dtype=knots_y.dtype,
            )
            utils.initialize_spline_knots_(knots_y, degree_y)

        return ctrl_points, ctrl_weights, knots_x, knots_y

    def _no_progressive_growing(self) -> bool:
        return self._interval < 1