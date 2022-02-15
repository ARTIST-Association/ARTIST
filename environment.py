import torch as th


class Sun_Distribution(object):
    def __init__(self, sun_configs, device):
        self.dist_type = sun_configs.DISTRIBUTION
        self.num_rays = sun_configs.GENERATE_N_RAYS

        dtype = th.get_default_dtype()
        if self.dist_type == "Normal":
            self.cfg = sun_configs.NORMAL_DIST
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

        elif self.dist_type == "Pillbox":
            raise ValueError("Not Implemented Yet")

    def sample(self, num_rays_on_hel):
        if self.dist_type == "Normal":
            distribution = th.distributions.MultivariateNormal(
                self.mean, self.cov)
            xi, yi = distribution.sample(
                (self.num_rays, num_rays_on_hel),
            ).transpose(0, 1).T
            return xi, yi


class Environment(object):
    def __init__(self, ambient_conditions_config, device):
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
        self.receiver_plane_x = self.cfg.RECEIVER.PLANE_X
        self.receiver_plane_y = self.cfg.RECEIVER.PLANE_Y
        self.receiver_resolution_x = self.cfg.RECEIVER.RESOLUTION_X
        self.receiver_resolution_y = self.cfg.RECEIVER.RESOLUTION_Y

        self.sun = Sun_Distribution(self.cfg.SUN, device)
