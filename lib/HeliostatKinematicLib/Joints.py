# system dependencies
import torch
import typing
import os
import sys

# local dependencies
module_dir = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(module_dir)
from ParametrizedModel import AbstractParametrizedModel

lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(lib_dir)
import CoordinateSystemsLib.ExtendedCoordinates as COORDS

class AbstractJoint(AbstractParametrizedModel):
    class Keys(typing.NamedTuple):
        cosys_pivot_east : str = 'cosys_pivot_east'
        cosys_pivot_north: str = 'cosys_pivot_north'
        cosys_pivot_up : str = 'cosys_pivot_up'
        east_tilt : str = 'east_tilt'
        north_tilt : str = 'north_tilt'
        up_tilt : str = 'up_tilt'
    keys = Keys()

    class DistKeys(typing.NamedTuple):
        cosys_pivot_azim : str = 'cosys_pivot_azim'
        cosys_pivot_elev : str = 'cosys_pivot_elev'
        cosys_pivot_rad : str = 'cosys_pivot_rad'
        east_tilt : str = 'east_tilt'
        north_tilt : str = 'north_tilt'
        up_tilt : str = 'up_tilt'
    dist_keys = DistKeys()

    # rotating directions
    east_rotation_direction : str = 'east'
    north_rotation_direction : str = 'north'
    up_rotation_direction : str = 'up'

    def __init__(self,
                 # parametrization
                 parameter_dict: typing.Dict[str, torch.Tensor] = {},
                 disturbance_dict: typing.Dict[str, torch.Tensor] = {},
                 
                 # parameter keys
                 cosys_pivot_east_key: typing.Optional[str] = None,
                 cosys_pivot_north_key: typing.Optional[str] = None,
                 cosys_pivot_up_key: typing.Optional[str] = None,
                 east_tilt_key: typing.Optional[str] = None,
                 north_tilt_key: typing.Optional[str] = None,
                 up_tilt_key: typing.Optional[str] = None,
                 
                 # disturbance keys
                 cosys_pivot_azim_dist_key: typing.Optional[str] = None,
                 cosys_pivot_elev_dist_key: typing.Optional[str] = None,
                 cosys_pivot_rad_dist_key: typing.Optional[str] = None,
                 east_tilt_dist_key: typing.Optional[str] = None,
                 north_tilt_dist_key: typing.Optional[str] = None,
                 up_tilt_dist_key: typing.Optional[str] = None,

                 # disturbance factors
                 dist_factor_rot: typing.Optional[torch.Tensor] = None,
                 dist_factor_len: typing.Optional[torch.Tensor] = None,

                 # pytorch defaults
                 dtype: torch.dtype = torch.get_default_dtype(),
                 device: torch.device = torch.device('cpu'),
                ):

        super().__init__(parameter_dict=parameter_dict,
                         disturbance_dict=disturbance_dict,
                         dist_factor_rot=dist_factor_rot,
                         dist_factor_len=dist_factor_len,
                         dtype=dtype,
                         device=device)

        # rotation direction
        self._rotation_direction : str = 'undefined'

        # parameter keys
        self.keys = AbstractJoint.Keys(
            cosys_pivot_east = cosys_pivot_east_key if cosys_pivot_east_key else AbstractJoint.keys.cosys_pivot_east,
            cosys_pivot_north = cosys_pivot_north_key if cosys_pivot_north_key else AbstractJoint.keys.cosys_pivot_north,
            cosys_pivot_up = cosys_pivot_up_key if cosys_pivot_up_key else AbstractJoint.keys.cosys_pivot_up,
            east_tilt = east_tilt_key if east_tilt_key else AbstractJoint.keys.east_tilt,
            north_tilt = north_tilt_key if north_tilt_key else AbstractJoint.keys.north_tilt,
            up_tilt = up_tilt_key if up_tilt_key else AbstractJoint.keys.up_tilt,
        )

        self.dist_keys = AbstractJoint.DistKeys(
            cosys_pivot_azim = cosys_pivot_azim_dist_key if cosys_pivot_azim_dist_key else AbstractJoint.dist_keys.cosys_pivot_azim,
            cosys_pivot_elev = cosys_pivot_elev_dist_key if cosys_pivot_elev_dist_key else AbstractJoint.dist_keys.cosys_pivot_elev,
            cosys_pivot_rad = cosys_pivot_rad_dist_key if cosys_pivot_rad_dist_key else AbstractJoint.dist_keys.cosys_pivot_rad,
            east_tilt = east_tilt_dist_key if east_tilt_dist_key else AbstractJoint.keys.dist_east_tilt,
            north_tilt = north_tilt_dist_key if north_tilt_dist_key else AbstractJoint.dist_keys.north_tilt,
            up_tilt = up_tilt_dist_key if up_tilt_dist_key else AbstractJoint.dist_keys.up_tilt,
        )
        # abstract class guard
        if type(self).__name__ == AbstractJoint.__name__:
                raise Exception("Don't implement an abstract class!")

    def cosysFromAngle(self, angle: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        # abstract class guard
        raise Exception("Abstract method must be overridden!")

    def rotationDirection(self):
        return self._rotation_direction

    ####################
    #-   Parameters   -#
    ####################
    def _cosys_origin(self):
        cosys_origin = self._parametrizedVector(parameter_keys=[self.keys.cosys_pivot_east, self.keys.cosys_pivot_north, self.keys.cosys_pivot_up],
                                                disturbance_keys=[self.dist_keys.cosys_pivot_azim, self.dist_keys.cosys_pivot_elev, self.dist_keys.cosys_pivot_rad])
        return cosys_origin

    def _east_tilt(self):
        east_tilt = self._rotParam(parameter_key=self.keys.east_tilt, disturbance_key=self.dist_keys.east_tilt)
        return east_tilt

    def _north_tilt(self):
        north_tilt = self._rotParam(parameter_key=self.keys.north_tilt, disturbance_key=self.dist_keys.north_tilt)
        return north_tilt

    def _up_tilt(self):
        up_tilt = self._rotParam(parameter_key=self.keys.up_tilt, disturbance_key=self.dist_keys.up_tilt)
        return up_tilt

    ###############
    #-   CoSys   -#
    ###############
    def _initialCosys(self):
        cosys = COORDS.translation4x4(trans_vec=self._cosys_origin(), device=self._device)
        cosys = cosys @ COORDS.eastRotation4x4(angle=self._east_tilt(), device=self._device)
        cosys = cosys @ COORDS.northRotation4x4(angle=self._north_tilt(), device=self._device)
        cosys = cosys @ COORDS.upRotation4x4(angle=self._up_tilt(), device=self._device)
        return cosys

    def cosysFromAngle(self, angle: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        cosys = self._initialCosys()
        return cosys

class FixedJoint(AbstractJoint):
    class Keys(typing.NamedTuple):
        cosys_pivot_east : str = 'fixed_cosys_pivot_east'
        cosys_pivot_north : str = 'fixed_cosys_pivot_north'
        cosys_pivot_up : str = 'fixed_cosys_pivot_up'
        east_tilt : str = 'fixed_cosys_east_tilt'
        north_tilt : str = 'fixed_cosys_north_tilt'
        up_tilt : str = 'fixed_cosys_up_tilt'
        cosys_pivot_azim_dist : str = 'fixed_cosys_pivot_azim_dist'
        cosys_pivot_elev_dist : str = 'fixed_cosys_pivot_elev_dist'
        cosys_pivot_rad_dist : str = 'cfixed_osys_pivot_rad_dist'
        east_tilt_dist : str = 'fixed_cosys_east_tilt_dist'
        north_tilt_dist : str = 'fixed_cosys_north_tilt_dist'
        up_tilt_dist_ : str = 'fixed_cosys_up_tilt_dist'
    keys = Keys()

    def __init__(self,
                 # parametrization
                 parameter_dict: typing.Dict[str, torch.Tensor] = {},
                 disturbance_dict: typing.Dict[str, torch.Tensor] = {},

                 # parameter keys
                 cosys_pivot_east_key: typing.Optional[str] = None,
                 cosys_pivot_north_key: typing.Optional[str] = None,
                 cosys_pivot_up_key: typing.Optional[str] = None,
                 east_tilt_key: typing.Optional[str] = None,
                 north_tilt_key: typing.Optional[str] = None,
                 up_tilt_key: typing.Optional[str] = None,
                 
                 # disturbance keys
                 cosys_pivot_azim_dist_key: typing.Optional[str] = None,
                 cosys_pivot_elev_dist_key: typing.Optional[str] = None,
                 cosys_pivot_rad_dist_key: typing.Optional[str] = None,
                 east_tilt_dist_key: typing.Optional[str] = None,
                 north_tilt_dist_key: typing.Optional[str] = None,
                 up_tilt_dist_key: typing.Optional[str] = None,

                 # disturbance factors
                 dist_factor_rot: typing.Optional[torch.Tensor] = None,
                 dist_factor_len: typing.Optional[torch.Tensor] = None,

                 # pytorch config
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.get_default_dtype(),
                 ):

        # optional parameters
        cosys_pivot_east_key = cosys_pivot_east_key if cosys_pivot_east_key else FixedJoint.keys.cosys_pivot_east
        cosys_pivot_north_key = cosys_pivot_north_key if cosys_pivot_north_key else FixedJoint.keys.cosys_pivot_north
        cosys_pivot_up_key = cosys_pivot_up_key if cosys_pivot_up_key else FixedJoint.keys.cosys_pivot_up
        east_tilt_key = east_tilt_key if east_tilt_key else FixedJoint.keys.east_tilt
        north_tilt_key = north_tilt_key if north_tilt_key else FixedJoint.keys.north_tilt
        up_tilt_key = up_tilt_key if up_tilt_key else FixedJoint.keys.up_tilt

        # optional disturbances
        cosys_pivot_azim_dist_key = cosys_pivot_azim_dist_key if cosys_pivot_azim_dist_key else FixedJoint.keys.cosys_pivot_azim_dist
        cosys_pivot_elev_dist_key = cosys_pivot_elev_dist_key if cosys_pivot_elev_dist_key else FixedJoint.keys.cosys_pivot_elev_dist
        cosys_pivot_rad_dist_key = cosys_pivot_rad_dist_key if cosys_pivot_rad_dist_key else FixedJoint.keys.cosys_pivot_rad_dist
        east_tilt_dist_key = east_tilt_dist_key if east_tilt_dist_key else FixedJoint.keys.east_tilt_dist
        north_tilt_dist_key = north_tilt_dist_key if north_tilt_dist_key else FixedJoint.keys.north_tilt_dist
        up_tilt_dist_key = up_tilt_dist_key if up_tilt_dist_key else FixedJoint.keys.up_tilt_dist

        # super init
        super().__init__(
                         parameter_dict=parameter_dict,
                         disturbance_dict=disturbance_dict,

                         # parameter keys
                         cosys_pivot_east_key=cosys_pivot_east_key,
                         cosys_pivot_north_key=cosys_pivot_north_key,
                         cosys_pivot_up_key=cosys_pivot_up_key,
                         east_tilt_key=east_tilt_key,
                         north_tilt_key=north_tilt_key,
                         up_tilt_key=up_tilt_key,

                         # disturbances
                         cosys_pivot_azim_dist_key=cosys_pivot_azim_dist_key,
                         cosys_pivot_elev_dist_key=cosys_pivot_elev_dist_key,
                         cosys_pivot_rad_dist_key=cosys_pivot_rad_dist_key,
                         east_tilt_dist_key=east_tilt_dist_key,
                         north_tilt_dist_key=north_tilt_dist_key,
                         up_tilt_dist_key=up_tilt_dist_key,

                         # disturbance factors
                         dist_factor_rot=dist_factor_rot,
                         dist_factor_len=dist_factor_len,

                         # pytorch config
                         device=device,
                         dtype=dtype
                         )
        
        # rotation direction
        self._rotation_direction : str = 'fixed'
    

class EastRotationJoint(AbstractJoint):
    class Keys(typing.NamedTuple):
        cosys_pivot_east : str          = 'east_rot_pivot_east'
        cosys_pivot_north : str         = 'east_rot_pivot_north'
        cosys_pivot_up : str            = 'east_rot_pivot_up'
        east_tilt : str                 = 'east_rot_east_tilt'
        north_tilt : str                = 'east_rot_north_tilt'
        up_tilt : str                   = 'east_rot_up_tilt'
    keys = Keys()

    class DistKeys(typing.NamedTuple):
        cosys_pivot_azim : str     = 'east_rot_cosys_pivot_azim'
        cosys_pivot_elev : str     = 'east_rot_cosys_pivot_elev'
        cosys_pivot_rad : str      = 'east_rot_cosys_pivot_rad'
        east_tilt : str            = 'east_rot_east_tilt'
        north_tilt : str           = 'east_rot_north_tilt'
        up_tilt : str              = 'east_rot_up_tilt'
    dist_keys = Keys()

    def __init__(self,
                 # parametrization
                 parameter_dict: typing.Dict[str, torch.Tensor] = {},
                 disturbance_dict: typing.Dict[str, torch.Tensor] = {},

                 # parameter keys
                 cosys_pivot_east_key: typing.Optional[str] = None,
                 cosys_pivot_north_key: typing.Optional[str] = None,
                 cosys_pivot_up_key: typing.Optional[str] = None,
                 east_tilt_key: typing.Optional[str] = None,
                 north_tilt_key: typing.Optional[str] = None,
                 up_tilt_key: typing.Optional[str] = None,
                 
                 # disturbance keys
                 cosys_pivot_azim_dist_key: typing.Optional[str] = None,
                 cosys_pivot_elev_dist_key: typing.Optional[str] = None,
                 cosys_pivot_rad_dist_key: typing.Optional[str] = None,
                 east_tilt_dist_key: typing.Optional[str] = None,
                 north_tilt_dist_key: typing.Optional[str] = None,
                 up_tilt_dist_key: typing.Optional[str] = None,

                 # disturbance factors
                 dist_factor_rot: typing.Optional[torch.Tensor] = None,
                 dist_factor_len: typing.Optional[torch.Tensor] = None,

                 # pytorch config
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.get_default_dtype(),
                 ):

        # optional parameters
        cosys_pivot_east_key = cosys_pivot_east_key if cosys_pivot_east_key else EastRotationJoint.keys.cosys_pivot_east
        cosys_pivot_north_key = cosys_pivot_north_key if cosys_pivot_north_key else EastRotationJoint.keys.cosys_pivot_north
        cosys_pivot_up_key = cosys_pivot_up_key if cosys_pivot_up_key else EastRotationJoint.keys.cosys_pivot_up
        east_tilt_key = east_tilt_key if east_tilt_key else EastRotationJoint.keys.east_tilt
        north_tilt_key = north_tilt_key if north_tilt_key else EastRotationJoint.keys.north_tilt
        up_tilt_key = up_tilt_key if up_tilt_key else EastRotationJoint.keys.up_tilt

        # optional disturbances
        cosys_pivot_azim_dist_key = cosys_pivot_azim_dist_key if cosys_pivot_azim_dist_key else EastRotationJoint.dist_keys.cosys_pivot_azim
        cosys_pivot_elev_dist_key = cosys_pivot_elev_dist_key if cosys_pivot_elev_dist_key else EastRotationJoint.dist_keys.cosys_pivot_elev
        cosys_pivot_rad_dist_key = cosys_pivot_rad_dist_key if cosys_pivot_rad_dist_key else EastRotationJoint.dist_keys.cosys_pivot_rad
        east_tilt_dist_key = east_tilt_dist_key if east_tilt_dist_key else EastRotationJoint.dist_keys.east_tilt
        north_tilt_dist_key = north_tilt_dist_key if north_tilt_dist_key else EastRotationJoint.dist_keys.north_tilt
        up_tilt_dist_key = up_tilt_dist_key if up_tilt_dist_key else EastRotationJoint.dist_keys.up_tilt

        # super init
        super().__init__(
                         parameter_dict=parameter_dict,
                         disturbance_dict=disturbance_dict,

                         # parameter keys
                         cosys_pivot_east_key=cosys_pivot_east_key,
                         cosys_pivot_north_key=cosys_pivot_north_key,
                         cosys_pivot_up_key=cosys_pivot_up_key,
                         east_tilt_key=east_tilt_key,
                         north_tilt_key=north_tilt_key,
                         up_tilt_key=up_tilt_key,

                         # disturbances
                         cosys_pivot_azim_dist_key=cosys_pivot_azim_dist_key,
                         cosys_pivot_elev_dist_key=cosys_pivot_elev_dist_key,
                         cosys_pivot_rad_dist_key=cosys_pivot_rad_dist_key,
                         east_tilt_dist_key=east_tilt_dist_key,
                         north_tilt_dist_key=north_tilt_dist_key,
                         up_tilt_dist_key=up_tilt_dist_key,

                         # disturbance factors
                         dist_factor_rot=dist_factor_rot,
                         dist_factor_len=dist_factor_len,

                         # pytorch config
                         device=device,
                         dtype=dtype
                         )
        
        # rotation direction
        self._rotation_direction : str = AbstractJoint.east_rotation_direction

    ###############
    #-   CoSys   -#
    ###############
    # override
    def _initialCosys(self):
        cosys = COORDS.translation4x4(trans_vec=self._cosys_origin(), dtype=self._dtype, device=self._device)
        cosys = cosys @ COORDS.northRotation4x4(angle=self._north_tilt(), dtype=self._dtype, device=self._device)
        cosys = cosys @ COORDS.upRotation4x4(angle=self._up_tilt(), dtype=self._dtype, device=self._device)
        return cosys

    # override
    def cosysFromAngle(self, angle: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        if not angle:
            angle = torch.tensor(0, dtype = self._dtype, device=self._device, requires_grad=False)

        cosys = self._initialCosys()
        cosys = cosys @ COORDS.eastRotation4x4(angle=angle + self._east_tilt(), dtype=self._dtype, device=self._device)
        return cosys

class NorthRotationJoint(AbstractJoint):
    class Keys(typing.NamedTuple):
        cosys_pivot_east : str = 'north_rot_pivot_east'
        cosys_pivot_north : str = 'north_rot_pivot_north'
        cosys_pivot_up : str = 'north_rot_pivot_up'
        east_tilt : str = 'north_rot_east_tilt'
        north_tilt : str = 'north_rot_north_tilt'
        up_tilt : str = 'north_rot_up_tilt'
    keys = Keys()  

    class DistKeys(typing.NamedTuple):
        cosys_pivot_azim : str = 'north_rot_cosys_pivot_azim'
        cosys_pivot_elev : str = 'north_rot_cosys_pivot_elev'
        cosys_pivot_rad : str = 'north_rot_cosys_pivot_rad'
        east_tilt : str = 'north_rot_east_tilt'
        north_tilt : str = 'north_rot_north_tilt'
        up_tilt : str = 'north_rot_up_tilt'
    dist_keys = DistKeys()   

    def __init__(self,
                 # parametrization
                 parameter_dict: typing.Dict[str, torch.Tensor] = {},
                 disturbance_dict: typing.Dict[str, torch.Tensor] = {},

                 # parameter keys
                 cosys_pivot_east_key: typing.Optional[str] = None,
                 cosys_pivot_north_key: typing.Optional[str] = None,
                 cosys_pivot_up_key: typing.Optional[str] = None,
                 east_tilt_key: typing.Optional[str] = None,
                 north_tilt_key: typing.Optional[str] = None,
                 up_tilt_key: typing.Optional[str] = None,
                 
                 # disturbance keys
                 cosys_pivot_azim_dist_key: typing.Optional[str] = None,
                 cosys_pivot_elev_dist_key: typing.Optional[str] = None,
                 cosys_pivot_rad_dist_key: typing.Optional[str] = None,
                 east_tilt_dist_key: typing.Optional[str] = None,
                 north_tilt_dist_key: typing.Optional[str] = None,
                 up_tilt_dist_key: typing.Optional[str] = None,

                 # disturbance factors
                 dist_factor_rot: typing.Optional[torch.Tensor] = None,
                 dist_factor_len: typing.Optional[torch.Tensor] = None,

                 # pytorch config
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.get_default_dtype(),
                 ):

        # optional parameters
        cosys_pivot_east_key = cosys_pivot_east_key if cosys_pivot_east_key else NorthRotationJoint.keys.cosys_pivot_east
        cosys_pivot_north_key = cosys_pivot_north_key if cosys_pivot_north_key else NorthRotationJoint.keys.cosys_pivot_north
        cosys_pivot_up_key = cosys_pivot_up_key if cosys_pivot_up_key else NorthRotationJoint.keys.cosys_pivot_up
        east_tilt_key = east_tilt_key if east_tilt_key else NorthRotationJoint.keys.east_tilt
        north_tilt_key = north_tilt_key if north_tilt_key else NorthRotationJoint.keys.north_tilt
        up_tilt_key = up_tilt_key if up_tilt_key else NorthRotationJoint.keys.up_tilt

        # optional disturbances
        cosys_pivot_azim_dist_key = cosys_pivot_azim_dist_key if cosys_pivot_azim_dist_key else NorthRotationJoint.dist_keys.cosys_pivot_azim
        cosys_pivot_elev_dist_key = cosys_pivot_elev_dist_key if cosys_pivot_elev_dist_key else NorthRotationJoint.dist_keys.cosys_pivot_elev
        cosys_pivot_rad_dist_key = cosys_pivot_rad_dist_key if cosys_pivot_rad_dist_key else NorthRotationJoint.dist_keys.cosys_pivot_rad
        east_tilt_dist_key = east_tilt_dist_key if east_tilt_dist_key else NorthRotationJoint.dist_keys.east_tilt
        north_tilt_dist_key = north_tilt_dist_key if north_tilt_dist_key else NorthRotationJoint.dist_keys.north_tilt
        up_tilt_dist_key = up_tilt_dist_key if up_tilt_dist_key else NorthRotationJoint.dist_keys.up_tilt

        super().__init__(
                         parameter_dict=parameter_dict,
                         disturbance_dict=disturbance_dict,

                         # parameter keys
                         cosys_pivot_east_key=cosys_pivot_east_key,
                         cosys_pivot_north_key=cosys_pivot_north_key,
                         cosys_pivot_up_key=cosys_pivot_up_key,
                         east_tilt_key=east_tilt_key,
                         north_tilt_key=north_tilt_key,
                         up_tilt_key=up_tilt_key,

                         # disturbances
                         cosys_pivot_azim_dist_key=cosys_pivot_azim_dist_key,
                         cosys_pivot_elev_dist_key=cosys_pivot_elev_dist_key,
                         cosys_pivot_rad_dist_key=cosys_pivot_rad_dist_key,
                         east_tilt_dist_key=east_tilt_dist_key,
                         north_tilt_dist_key=north_tilt_dist_key,
                         up_tilt_dist_key=up_tilt_dist_key,

                         # disturbance factors
                         dist_factor_rot=dist_factor_rot,
                         dist_factor_len=dist_factor_len,

                         # pytorch config
                         device=device,
                         dtype=dtype
                         )

        # rotation direction
        self._rotation_direction : str = AbstractJoint.north_rotation_direction

    ###############
    #-   CoSys   -#
    ###############
    # override
    def _initialCosys(self):
        cosys = COORDS.translation4x4(trans_vec=self._cosys_origin(), dtype=self._dtype, device=self._device)
        cosys = cosys @ COORDS.eastRotation4x4(angle=self._east_tilt(), dtype=self._dtype, device=self._device)
        cosys = cosys @ COORDS.upRotation4x4(angle=self._up_tilt(), dtype=self._dtype, device=self._device)
        return cosys

    # override
    def cosysFromAngle(self, angle: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        if not angle:
            angle = torch.tensor(0, dtype = self._dtype, device=self._device, requires_grad=False)

        cosys = self._initialCosys()
        cosys = cosys @ COORDS.northRotation4x4(angle=angle + self._north_tilt(), dtype=self._dtype, device=self._device)
        return cosys

class UpRotationJoint(AbstractJoint):
    class Keys(typing.NamedTuple):
        cosys_pivot_east : str = 'up_rot_pivot_east'
        cosys_pivot_north : str = 'up_rot_pivot_north'
        cosys_pivot_up : str = 'up_rot_pivot_up'
        east_tilt : str = 'up_rot_east_tilt'
        north_tilt : str = 'up_rot_north_tilt'
        up_tilt : str = 'up_rot_up_tilt'
    keys = Keys()

    class DistKeys(typing.NamedTuple):
        cosys_pivot_azim : str = 'up_rot_cosys_pivot_azim'
        cosys_pivot_elev : str = 'up_rot_cosys_pivot_elev'
        cosys_pivot_rad : str = 'up_rot_cosys_pivot_rad'
        east_tilt : str = 'up_rot_east_tilt'
        north_tilt : str = 'up_rot_north_tilt'
        up_tilt : str = 'up_rot_up_tilt'
    dist_keys = DistKeys()

    

    def __init__(self,
                 # parametrization
                 parameter_dict: typing.Dict[str, torch.Tensor] = {},
                 disturbance_dict: typing.Dict[str, torch.Tensor] = {},

                 # parameter keys
                 cosys_pivot_east_key: typing.Optional[str] = None,
                 cosys_pivot_north_key: typing.Optional[str] = None,
                 cosys_pivot_up_key: typing.Optional[str] = None,
                 east_tilt_key: typing.Optional[str] = None,
                 north_tilt_key: typing.Optional[str] = None,
                 up_tilt_key: typing.Optional[str] = None,
                 
                 # disturbance keys
                 cosys_pivot_azim_dist_key: typing.Optional[str] = None,
                 cosys_pivot_elev_dist_key: typing.Optional[str] = None,
                 cosys_pivot_rad_dist_key: typing.Optional[str] = None,
                 east_tilt_dist_key: typing.Optional[str] = None,
                 north_tilt_dist_key: typing.Optional[str] = None,
                 up_tilt_dist_key: typing.Optional[str] = None,

                 # disturbance factors
                 dist_factor_rot: typing.Optional[torch.Tensor] = None,
                 dist_factor_len: typing.Optional[torch.Tensor] = None,

                 # pytorch config
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.get_default_dtype(),
                 ):

        # optional parameters
        cosys_pivot_east_key = cosys_pivot_east_key if cosys_pivot_east_key else UpRotationJoint.keys.cosys_pivot_east
        cosys_pivot_north_key = cosys_pivot_north_key if cosys_pivot_north_key else UpRotationJoint.keys.cosys_pivot_north
        cosys_pivot_up_key = cosys_pivot_up_key if cosys_pivot_up_key else UpRotationJoint.keys.cosys_pivot_up
        east_tilt_key = east_tilt_key if east_tilt_key else UpRotationJoint.keys.east_tilt
        north_tilt_key = north_tilt_key if north_tilt_key else UpRotationJoint.keys.north_tilt
        up_tilt_key = up_tilt_key if up_tilt_key else UpRotationJoint.keys.up_tilt

        # optional disturbances
        cosys_pivot_azim_dist_key = cosys_pivot_azim_dist_key if cosys_pivot_azim_dist_key else UpRotationJoint.dist_keys.cosys_pivot_azim
        cosys_pivot_elev_dist_key = cosys_pivot_elev_dist_key if cosys_pivot_elev_dist_key else UpRotationJoint.dist_keys.cosys_pivot_elev
        cosys_pivot_rad_dist_key = cosys_pivot_rad_dist_key if cosys_pivot_rad_dist_key else UpRotationJoint.dist_keys.cosys_pivot_rad
        east_tilt_dist_key = east_tilt_dist_key if east_tilt_dist_key else UpRotationJoint.dist_keys.east_tilt
        north_tilt_dist_key = north_tilt_dist_key if north_tilt_dist_key else UpRotationJoint.dist_keys.north_tilt
        up_tilt_dist_key = up_tilt_dist_key if up_tilt_dist_key else UpRotationJoint.dist_keys.up_tilt

        super().__init__(
                         parameter_dict=parameter_dict,
                         disturbance_dict=disturbance_dict,

                         # parameter keys
                         cosys_pivot_east_key=cosys_pivot_east_key,
                         cosys_pivot_north_key=cosys_pivot_north_key,
                         cosys_pivot_up_key=cosys_pivot_up_key,
                         east_tilt_key=east_tilt_key,
                         north_tilt_key=north_tilt_key,
                         up_tilt_key=up_tilt_key,

                         # disturbances
                         cosys_pivot_azim_dist_key=cosys_pivot_azim_dist_key,
                         cosys_pivot_elev_dist_key=cosys_pivot_elev_dist_key,
                         cosys_pivot_rad_dist_key=cosys_pivot_rad_dist_key,
                         east_tilt_dist_key=east_tilt_dist_key,
                         north_tilt_dist_key=north_tilt_dist_key,
                         up_tilt_dist_key=up_tilt_dist_key,

                         # disturbance factors
                         dist_factor_rot=dist_factor_rot,
                         dist_factor_len=dist_factor_len,

                         # pytorch config
                         device=device,
                         dtype=dtype
                         )

        # rotation_direction
        self._rotation_direction : str = AbstractJoint.up_rotation_direction

    ###############
    #-   CoSys   -#
    ###############
    # override
    def _initialCosys(self):
        cosys = COORDS.translation4x4(trans_vec=self._cosys_origin(), dtype=self._dtype, device=self._device)
        cosys = cosys @ COORDS.eastRotation4x4(angle=self._east_tilt(), dtype=self._dtype, device=self._device)
        cosys = cosys @ COORDS.northRotation4x4(angle=self._north_tilt(), dtype=self._dtype, device=self._device)
        return cosys

    # override
    def cosysFromAngle(self, angle: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        if not angle:
            angle = torch.tensor(0, dtype = self._dtype, device=self._device, requires_grad=False)

        cosys = self._initialCosys()
        cosys = cosys @ COORDS.upRotation4x4(angle=angle + self._up_tilt(), dtype=self._dtype, device=self._device)
        return cosys