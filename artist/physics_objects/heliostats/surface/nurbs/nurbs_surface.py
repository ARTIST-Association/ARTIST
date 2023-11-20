
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