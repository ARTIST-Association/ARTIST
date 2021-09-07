import torch as th

class Sun_Distribution(object):
    def __init__(self, sun_configs, device):
        self.dist_type = sun_configs.DISTRIBUTION
        self.num_rays  = sun_configs.GENERATE_N_RAYS
        
        if self.dist_type == "Normal":
            self.cfg = sun_configs.NORMAL_DIST
            self.mean = th.tensor(self.cfg.MEAN, dtype=th.float32, device=device)
            self.cov =  th.tensor(self.cfg.COV, dtype=th.float32, device=device)
            
        if self.dist_type == "Pillbox":
            raise ValueError("Not Implemented Yet")
            
    def sample_(self):
        if self.dist_type == "Normal":
            xi, yi = th.distributions.MultivariateNormal(self.mean, self.cov).sample((self.num_rays,)).T # TODO Überprüfen ob hier .T hinmuss
            return xi, yi




class Environment(object):
    def __init__(self, ambient_conditions_config, device):
        self.cfg = ambient_conditions_config
        
        self.receiver_center                    = th.tensor(self.cfg.RECEIVER.CENTER, dtype=th.float32, device = device)
        self.receiver_plane_normal              = th.tensor(self.cfg.RECEIVER.PLANE_NORMAL, dtype=th.float32, device = device) 
        self.receiver_plane_x                   = self.cfg.RECEIVER.PLANE_X
        self.receiver_plane_y                   = self.cfg.RECEIVER.PLANE_Y 
        self.receiver_resolution_x              = self.cfg.RECEIVER.RESOLUTION_X
        self.receiver_resolution_y              = self.cfg.RECEIVER.RESOLUTION_Y
        
        sun_origin                              = th.tensor(self.cfg.SUN.ORIGIN, dtype=th.float32, device = device)
        self.sun_origin                         = sun_origin/th.linalg.norm(sun_origin) 
        self.Sun                                = Sun_Distribution(self.cfg.SUN, device)
        
        
        

        
        