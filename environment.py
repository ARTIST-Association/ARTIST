from typing import Tuple

import torch
import torch as th
from yacs.config import CfgNode


class Sun_Distribution(object):
    def __init__(self, sun_configs: CfgNode, device: th.device) -> None:
        self.dist_type: str = sun_configs.DISTRIBUTION
        self.num_rays: int = sun_configs.GENERATE_N_RAYS

        dtype = th.get_default_dtype()
        if self.dist_type == "Normal":
            self.cfg: CfgNode = sun_configs.NORMAL_DIST
            self.mean = th.tensor(
                self.cfg.MEAN,
                dtype=dtype,
                device=device,
            )
            self.cov = th.tensor(
                self.cfg.COV,
                dtype=dtype,
                device=device,
            )
            self.distribution = th.distributions.MultivariateNormal(
                self.mean, self.cov)

        elif self.dist_type == "Pillbox":
            raise ValueError("Not Implemented Yet")
        else:
            raise ValueError("unknown sun distribution type")

    def sample(
            self,
            num_rays_on_hel: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.dist_type == "Normal":
            xi, yi = self.distribution.sample(
                (self.num_rays, num_rays_on_hel),
            ).transpose(0, 1).T
            return xi, yi
        else:
            raise ValueError('unknown sun distribution type')


class Environment(object):
    def __init__(
            self,
            ambient_conditions_config: CfgNode,
            device: th.device,
    ) -> None:
        self.cfg = ambient_conditions_config
        dtype = th.get_default_dtype()

        self.receiver_center = th.tensor(
            self.cfg.RECEIVER.CENTER,
            dtype=dtype,
            device=device,
        )
        self.receiver_plane_normal = th.tensor(
            self.cfg.RECEIVER.PLANE_NORMAL,
            dtype=dtype,
            device=device,
        )
        self.receiver_plane_x: float = self.cfg.RECEIVER.PLANE_X
        self.receiver_plane_y: float = self.cfg.RECEIVER.PLANE_Y
        self.receiver_resolution_x: int = self.cfg.RECEIVER.RESOLUTION_X
        self.receiver_resolution_y: int = self.cfg.RECEIVER.RESOLUTION_Y

        self.sun = Sun_Distribution(self.cfg.SUN, device)

