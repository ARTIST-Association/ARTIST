# system dependencies
import sys
import os
import typing
import torch
import pickle

# local dependencies
module_dir = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(module_dir)
import Actuators
import Joints
import AlignmentDisturbanceModel as DM

lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(lib_dir)
import CoordinateSystemsLib.ExtendedCoordinates as COORDS
from ParametrizedModel import AbstractParametrizedModel
from HeliostatTrainingLib.HeliostatDatapoint import HeliostatDataPoint

def alignmentDeviationRadFromNormal(normal: torch.tensor, normal_target: torch.tensor) -> torch.tensor:
    cos_delta = torch.dot(normal, normal_target) / (normal.norm() * normal_target.norm())
    delta = torch.abs(torch.arccos(cos_delta))

    # for extremely small angles torch.arccos might converge to nan
    # if so the norm of the vector difference is used instead
    if torch.isnan(delta) and not torch.any(torch.isnan(normal)) and not torch.any(torch.isnan(normal_target)):
        normal_diff = normal - normal_target
        delta = normal_diff.norm()

    # asserts(not torch.isnan(delta)),'delta: ' + str(delta) + ' normal: ' + str(normal) + ' normal_target: ' + str(normal_target)
    return delta

def targetNormalFromSourcePos(aimpoint: torch.tensor, source_pos: torch.tensor, pivoting_point: torch.tensor):
    to_source = source_pos - pivoting_point
    to_source = to_source / to_source.norm()
    return targetNormalFromSourceVec(aimpoint=aimpoint, to_source=to_source, pivoting_point=pivoting_point)

def targetNormalFromSourceVec(aimpoint: torch.tensor, to_source: torch.tensor, pivoting_point: torch.tensor):
    aim_vec = aimpoint - pivoting_point
    aim_vec = aim_vec / aim_vec.norm()
    target_normal = to_source + aim_vec
    target_normal = target_normal / target_normal.norm()
    return target_normal

class AbstractAlignmentModel(AbstractParametrizedModel):
    class Keys(typing.NamedTuple):
        position_east : str        = 'position_east'
        position_north : str       = 'position_north'
        position_up : str          = 'position_up'
    keys = Keys()

    class DistKeys(typing.NamedTuple):
        position_azim : str   = 'position_azim'
        position_elev : str   = 'position_elev'
        position_rad : str    = 'position_rad'
    dist_keys = DistKeys()

    def __init__(self,
                 # parametrizations
                 parameter_dict : typing.Dict[str, torch.Tensor],
                 disturbance_dict : typing.Dict[str, torch.Tensor],

                 # disturbance parameters
                 dist_factor_rot: typing.Optional[torch.Tensor] = None,
                 dist_factor_len: typing.Optional[torch.Tensor] = None,
                 dist_factor_perc: typing.Optional[torch.Tensor] = None,

                 # parameter keys
                 position_east_key : typing.Optional[str] = None,
                 position_north_key : typing.Optional[str] = None,
                 position_up_key : typing.Optional[str] = None,

                 # disturbance keys
                 position_azim_dist_key : typing.Optional[str] = None,
                 position_elev_dist_key : typing.Optional[str] = None,
                 position_rad_dist_key : typing.Optional[str] = None,

                 # pytorch config
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.get_default_dtype(),
                ):

        super().__init__(
                         parameter_dict=parameter_dict,
                         disturbance_dict=disturbance_dict,
                         dist_factor_rot=dist_factor_rot,
                         dist_factor_len=dist_factor_len,
                         dist_factor_perc=dist_factor_perc,
                         dtype=dtype,
                         device=device
                         )

        # optional keys
        self.keys = AbstractAlignmentModel.Keys(
            position_east = position_east_key if position_east_key else AbstractAlignmentModel.keys.position_east,
            position_north = position_north_key if position_north_key else AbstractAlignmentModel.keys.position_north,
            position_up = position_up_key if position_up_key else AbstractAlignmentModel.keys.position_up,
        )
        self.dist_keys = AbstractAlignmentModel.DistKeys(
            position_azim = position_azim_dist_key if position_azim_dist_key else AbstractAlignmentModel.dist_keys.position_azim,
            position_elev = position_elev_dist_key if position_elev_dist_key else AbstractAlignmentModel.dist_keys.position_elev,
            position_rad = position_rad_dist_key if position_rad_dist_key else AbstractAlignmentModel.dist_keys.position_rad,
        )

        # abstract class guard
        if type(self).__name__ == AbstractParametrizedModel.__name__:
                raise Exception("Don't implement an abstract class!")
    
    def alignmentModelPath(self, dir_path: str) -> str:
        model_path = os.path.join(dir_path, "alignment.model")
        return model_path

    def loadAlignmentModel(self, dir_path: str):
        file_path = self.alignmentModelPath(dir_path=dir_path)
        file = open(file_path, 'rb')
        self = pickle.load(file)
        file.close()
        return self

    def saveAlignmentModel(self, dir_path: str):
        file_path = self.alignmentModelPath(dir_path=dir_path)
        file = open(file_path, 'wb')
        pickle.dump(self, file)
        file.close()

    def summary(self) -> str:
        summary_str = "Alignment Model of type:\n"
        return summary_str

    ######################
    #-   Normalization   -#
    ######################
    def _maxActuatorSteps(self):
        # abstract class guard
        raise Exception("Abstract method must be overridden!")

    def _normalizeActuatorSteps(self, actuator_steps: torch.Tensor):
        div_tensor = torch.tensor([self._maxActuatorSteps(), self._maxActuatorSteps()], dtype=self._dtype, device=self._device)
        normed_vec = actuator_steps / div_tensor
        return normed_vec

    def _normalizeEnvState(self, env_state: typing.Optional[typing.Dict[str,torch.Tensor]] = None):
        # ToDo normalize values here:
        normalized_env_state = None

        if env_state:
            normalized_env_state = {}
            for k, e_s in env_state.items():
                if torch.is_tensor(e_s):
                    if k == HeliostatDataPoint.env_keys.month:
                        normalized_env_state[k + '_sin'] = torch.sin(e_s / 12 * 2 * torch.pi)
                        normalized_env_state[k + '_cos'] = torch.cos(e_s / 12 * 2 * torch.pi)
                    elif k == HeliostatDataPoint.env_keys.elapsed_days:
                        normalized_env_state[k + '_sin'] = torch.sin(e_s * 2 * torch.pi)
                        normalized_env_state[k + '_cos'] = torch.cos(e_s * 2 * torch.pi)
                    else:
                        normalized_env_state[k] = e_s

            # not implemented guard
            # raise Exception("Method not yet implemented!")
            

        return normalized_env_state

    ####################
    #-   Parameters   -#
    ####################
    def _position(self) -> torch.Tensor:
        position = self._parametrizedVector(parameter_keys=[self.keys.position_east, self.keys.position_north, self.keys.position_up],
                                            disturbance_keys=[self.dist_keys.position_azim, self.dist_keys.position_elev, self.dist_keys.position_rad],
                                            )
        return position

    ###############
    #-   CoSys   -#
    ###############
    def _normalFromCosys(self, cosys: torch.Tensor) -> torch.Tensor:
        vec = torch.tensor([0, -1, 0, 0], dtype=self._dtype, device=self._device)
        normal = (cosys @ vec)[:3]
        return normal

    def _pivotingPointFromCosys(self, cosys: torch.Tensor) -> torch.Tensor:
        vec = torch.tensor([0, 0, 0, 1], dtype=self._dtype, device=self._device)
        pivoting_point = (cosys @ vec)[:3]
        return pivoting_point

    def _sideEastFromCosys(self, cosys: torch.Tensor) -> torch.Tensor:
        vec = torch.tensor([1, 0, 0, 0], dtype=self._dtype, device=self._device)
        side_east = (cosys @ vec)[:3]
        return side_east

    def _sideUpFromCosys(self, cosys: torch.Tensor) -> torch.Tensor:
        vec = torch.tensor([0, 0, 1, 0], dtype=self._dtype, device=self._device)
        side_up = (cosys @ vec)[:3]
        return side_up

    ############################
    #-   Forward Kinematics   -#
    ############################
    def alignmentFromActuatorSteps(self, actuator_steps: torch.Tensor, env_state: typing.Optional[typing.Dict[str,torch.Tensor]] = None) -> torch.Tensor :
        
        # abstract class guard
        raise Exception("Abstract method must be overridden!")

        # return normal, pivoting_point, side_east, side_up, actuator_steps, cosys

    ############################
    #-   Inverse Kinematics   -#
    ############################
    def alignmentFromSourceVec(self, 
                               to_source: torch.Tensor, 
                               aimpoint: torch.Tensor,
                               max_num_epochs: int = 20,
                               eps: torch.Tensor = torch.tensor(10**(-12)),
                               ) -> typing.List[torch.Tensor]: 
        # abstract class guard
        raise Exception("Abstract method must be overridden!")

    def alignmentFromSourcePos(self, 
                               source_pos: torch.Tensor, 
                               aimpoint: torch.Tensor,
                               max_num_epochs: int = 20,
                               eps: torch.Tensor = torch.tensor(10**(-12)),
                               ) -> typing.List[torch.Tensor]:
        # abstract class guard
        raise Exception("Abstract method must be overridden!")
        
    ###################
    #-   Alignment   -#
    ###################
    def alignFromDataPoint(self, data_point: HeliostatDataPoint) -> typing.Dict[str, torch.Tensor]:
        env_state = data_point.env_state()
        alignment_error = None
        normal_target = None
        if data_point.ax1_steps and data_point.ax2_steps:
            actuator_steps = torch.tensor([data_point.ax1_steps, data_point.ax2_steps], dtype=self._dtype, device=self._device)
            # normal, pivoting_point, side_east, side_up, actuator_steps, cosys = self.alignmentFromActuatorSteps(actuator_steps=actuator_steps, env_state=data_point.env_state())
            normal, pivoting_point, side_east, side_up, actuator_steps, cosys = self.alignmentFromActuatorSteps(actuator_steps=actuator_steps, env_state=env_state)
        
            if isinstance(data_point.aimpoint, torch.Tensor) and isinstance(data_point.source_pos, torch.Tensor):
                normal_target = targetNormalFromSourcePos(aimpoint=data_point.aimpoint, source_pos=data_point.source_pos, pivoting_point=pivoting_point)
                alignment_error = alignmentDeviationRadFromNormal(normal=normal, normal_target=normal_target)
            elif isinstance(data_point.aimpoint, torch.Tensor) and isinstance(data_point.to_source, torch.Tensor):
                normal_target = targetNormalFromSourceVec(aimpoint=data_point.aimpoint, to_source=data_point.to_source, pivoting_point=pivoting_point)
                alignment_error = alignmentDeviationRadFromNormal(normal=normal, normal_target=normal_target)
            
        elif isinstance(data_point.aimpoint, torch.Tensor) and isinstance(data_point.source_pos, torch.Tensor):
            normal, pivoting_point, side_east, side_up, actuator_steps, cosys = self.alignmentFromSourcePos(source_pos=data_point.source_pos, aimpoint=data_point.aimpoint)
        elif isinstance(data_point.aimpoint, torch.Tensor) and isinstance(data_point.to_source, torch.Tensor):
            normal, pivoting_point, side_east, side_up, actuator_steps, cosys = self.alignmentFromSourceVec(to_source=data_point.to_source, aimpoint=data_point.aimpoint)
            
        alignment_dict = {
            'normal' : normal,
            'normal_target' : normal_target,
            'pivoting_point' : pivoting_point,
            'side_east' : side_east,
            'side_up' : side_up,
            'actuator_steps' : actuator_steps,
            'cosys' : torch.stack([side_east, side_up, normal]),
            HeliostatDataPoint.training_result_keys.alignment_deviation : alignment_error,
            }

        # for value in alignment_dict.values():
        #     assert(not torch.any(torch.isnan(value))), 'alignment_dict:\n' + str(alignment_dict) + '\ndisturbances:\n' + str(self._disturbance_dict) + '\n' + str(self._disturbance_dict)
        
        return alignment_dict
    
    ##################
    #-   Training   -#
    ##################
    def getTrainingParams(self) -> typing.List[torch.Tensor]:
        # abstract class guard
        raise Exception("Abstract method must be overridden!")

class AbstractAlignmentModelWithDisturbanceModel(AbstractAlignmentModel):
    def __init__(self,
                 
                 # parametrization
                 parameter_dict : typing.Dict[str, torch.Tensor],
                 disturbance_model : typing.Optional[DM.AbstractAlignmentDisturbanceModel] = None,

                 # parameter keys
                 position_east_key : typing.Optional[str] = None,
                 position_north_key : typing.Optional[str] = None,
                 position_up_key : typing.Optional[str] = None,

                 # disturbance keys
                 position_azim_dist_key : typing.Optional[str] = None,
                 position_elev_dist_key : typing.Optional[str] = None,
                 position_rad_dist_key : typing.Optional[str] = None,

                 # disturbance factors
                 dist_factor_rot: typing.Optional[torch.Tensor] = None,
                 dist_factor_len: typing.Optional[torch.Tensor] = None,

                 # pytorch config
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.get_default_dtype(),
                ):
        self._disturbance_model = disturbance_model
        disturbance_dict = {}

        super().__init__(
                         parameter_dict=parameter_dict,
                         disturbance_dict=disturbance_dict,
                         position_east_key=position_east_key,
                         position_north_key=position_north_key,
                         position_up_key=position_up_key,
                         position_azim_dist_key=position_azim_dist_key,
                         position_elev_dist_key=position_elev_dist_key,
                         position_rad_dist_key=position_rad_dist_key,
                         dist_factor_rot=dist_factor_rot,
                         dist_factor_len=dist_factor_len,
                         dtype=dtype,
                         device=device
                         )

    def fixateDisturbances(self) -> typing.Dict[str, torch.Tensor]:
        actuator_steps = torch.zeros(2, dtype=self._dtype, device=self._device)
        self._disturbance_dict = self.predictDisturbances(actuator_steps=actuator_steps)
        for key, item in self._disturbance_dict.items():
            if key[-4:] == 'tilt' or key[-5:] == 'angle':
                self._parameter_dict[key] = self._rotParam(parameter_key=key, disturbance_key=key)
            else:
                self._parameter_dict[key] = self._percParam(parameter_key=key, disturbance_key=key)
            # elif key in self.keys:
            #     if key[-4:] == 'tilt' or key[-5:] == 'angle':
            #         self._parameter_dict[key] = item / 1000
            #     else:
            #         self._parameter_dict[key] = item
        return self._disturbance_dict        

    def loadAlignmentModel(self, dir_path: str):
        self = super().loadAlignmentModel(dir_path=dir_path)
        if self._disturbance_model:
            self._disturbance_model = self._disturbance_model.loadDisturbanceModel(dir_path=dir_path)
        return self

    def saveAlignmentModel(self, dir_path: str):
        super().saveAlignmentModel(dir_path=dir_path)
        if self._disturbance_model:
            self._disturbance_model.saveDisturbanceModel(dir_path=dir_path)

    def predictDisturbances(self, actuator_steps: torch.Tensor, env_state: typing.Optional[typing.Dict[str,torch.Tensor]] = None) -> typing.Dict[str, torch.Tensor]:
        disturbance_dict = {}
        if self._disturbance_model:
            normalized_actuator_steps = self._normalizeActuatorSteps(actuator_steps=actuator_steps)
            normalized_env_state = self._normalizeEnvState(env_state=env_state)

            disturbance_dict = self._disturbance_model.predictDisturbances(normalized_actuator_steps=normalized_actuator_steps, normalized_env_state=normalized_env_state)
        
        self._setDisturbances(disturbance_dict=disturbance_dict)
        return disturbance_dict

    def _setDisturbances(self, disturbance_dict: typing.Dict[str,torch.Tensor]):
        self.setDisturbanceDict(disturbance_dict=disturbance_dict)
        
    ##################
    #-   Training   -#
    ##################
    def getTrainingParams(self) -> typing.List[torch.Tensor]:
        return self._disturbance_model.modelParameters()

class PointAlignmentModel(AbstractAlignmentModelWithDisturbanceModel):
    class Keys(typing.NamedTuple):
        position_east : str                         = 'position_east'
        position_north : str                        = 'position_north'
        position_up : str                           = 'position_up'

        # concentrator keys
        concentrator_cosys_pivot_east : str     = 'concentrator_cosys_pivot_azim'
        concentrator_cosys_pivot_north : str     = 'concentrator_cosys_pivot_elev'
        concentrator_cosys_pivot_up : str      = 'concentrator_cosys_pivot_rad'
        concentrator_east_tilt : str            = 'concentrator_east_tilt'
        concentrator_north_tilt : str           = 'concentrator_north_tilt'
        concentrator_up_tilt : str              = 'concentrator_up_tilt'

    keys = Keys()

    class DistKeys(typing.NamedTuple):
        # concentrator keys
        concentrator_cosys_pivot_azim : str     = 'concentrator_cosys_pivot_azim'
        concentrator_cosys_pivot_elev : str     = 'concentrator_cosys_pivot_elev'
        concentrator_cosys_pivot_rad : str      = 'concentrator_cosys_pivot_rad'
        concentrator_east_tilt : str            = 'concentrator_east_tilt'
        concentrator_north_tilt : str           = 'concentrator_north_tilt'
        concentrator_up_tilt : str              = 'concentrator_up_tilt'
    dist_keys = DistKeys()

    def __init__(self,
                 # parametrization
                 parameter_dict : typing.Optional[typing.Dict[str, torch.Tensor]] = None,

                 position: typing.Optional[torch.Tensor] = None,

                 # parameter keys
                 position_east_key : typing.Optional[str] = None,
                 position_north_key : typing.Optional[str] = None,
                 position_up_key : typing.Optional[str] = None,

                 # disturbance keys
                 position_azim_dist_key : typing.Optional[str] = None,
                 position_elev_dist_key : typing.Optional[str] = None,
                 position_rad_dist_key : typing.Optional[str] = None,

                 # disturbance model
                 disturbance_model : typing.Optional[DM.AbstractAlignmentDisturbanceModel] = None,

                 # disturbance factors
                 dist_factor_rot: typing.Optional[torch.Tensor] = None,
                 dist_factor_len: typing.Optional[torch.Tensor] = None,

                 # pytorch config
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.get_default_dtype(),
                ):

        self.keys = PointAlignmentModel.Keys()
        self.dist_keys = PointAlignmentModel.DistKeys()

        if not parameter_dict:
            parameter_dict = {
                                # position
                                PointAlignmentModel.keys.position_east                            : position[COORDS.E],
                                PointAlignmentModel.keys.position_north                           : position[COORDS.N],
                                PointAlignmentModel.keys.position_up                              : position[COORDS.U],
            }

        super().__init__(
                         parameter_dict=parameter_dict,
                         disturbance_model=disturbance_model,
                         position_east_key=position_east_key,
                         position_north_key=position_north_key,
                         position_up_key=position_up_key,
                         position_azim_dist_key=position_azim_dist_key,
                         position_elev_dist_key=position_elev_dist_key,
                         position_rad_dist_key=position_rad_dist_key,
                         dist_factor_rot=dist_factor_rot,
                         dist_factor_len=dist_factor_len,
                         dtype=dtype,
                         device=device
                         )

        self._concentrator_cosys = Joints.FixedJoint(   cosys_pivot_east_key=        PointAlignmentModel.keys.concentrator_cosys_pivot_east,
                                                        cosys_pivot_north_key=       PointAlignmentModel.keys.concentrator_cosys_pivot_north,
                                                        cosys_pivot_up_key=          PointAlignmentModel.keys.concentrator_cosys_pivot_up,
                                                        east_tilt_key=               PointAlignmentModel.keys.concentrator_east_tilt,
                                                        north_tilt_key=              PointAlignmentModel.keys.concentrator_north_tilt,
                                                        up_tilt_key=                 PointAlignmentModel.keys.concentrator_up_tilt,
                                                        cosys_pivot_azim_dist_key=   PointAlignmentModel.dist_keys.concentrator_cosys_pivot_azim,
                                                        cosys_pivot_elev_dist_key=   PointAlignmentModel.dist_keys.concentrator_cosys_pivot_elev,
                                                        cosys_pivot_rad_dist_key=    PointAlignmentModel.dist_keys.concentrator_cosys_pivot_rad,
                                                        east_tilt_dist_key=          PointAlignmentModel.dist_keys.concentrator_east_tilt,
                                                        north_tilt_dist_key=         PointAlignmentModel.dist_keys.concentrator_north_tilt,
                                                        up_tilt_dist_key=            PointAlignmentModel.dist_keys.concentrator_up_tilt,

                                                        parameter_dict=              parameter_dict,
                                                        dist_factor_rot=             dist_factor_rot,
                                                        dist_factor_len=             dist_factor_len,

                                                        device=                      self._device,
                                                        dtype =                      self._dtype
        )

    # override
    def _setDisturbances(self, disturbance_dict: typing.Dict[str, torch.Tensor]):
        super()._setDisturbances(disturbance_dict=disturbance_dict)
        self._concentrator_cosys.setDisturbanceDict(disturbance_dict=disturbance_dict)

    # override
    def fixateDisturbances(self) -> typing.Dict[str, torch.Tensor]:
        disturbance_dict = super().fixateDisturbances()
        if self.dist_keys.position_rad in disturbance_dict:
            pos_offset = self._parametrizedVector(parameter_keys = [None], disturbance_keys=[self.dist_keys.position_azim, self.dist_keys.position_elev, self.dist_keys.position_rad])
            self._parameter_dict[self.keys.position_east] += pos_offset[COORDS.E]
            self._parameter_dict[self.keys.position_north] += pos_offset[COORDS.N]
            self._parameter_dict[self.keys.position_up] += pos_offset[COORDS.U]

        if self.dist_keys.concentrator_cosys_pivot_rad in disturbance_dict:
            conc_offset = self._parametrizedVector(parameter_keys = [None], disturbance_keys=[self.dist_keys.concentrator_cosys_pivot_azim, self.dist_keys.concentrator_cosys_pivot_elev, self.dist_keys.concentrator_cosys_pivot_rad])
            self._parameter_dict[self.keys.concentrator_cosys_pivot_east] += conc_offset[COORDS.E]
            self._parameter_dict[self.keys.concentrator_cosys_pivot_north] += conc_offset[COORDS.N]
            self._parameter_dict[self.keys.concentrator_cosys_pivot_up] += conc_offset[COORDS.U]

    ######################
    #-   Normalization   -#
    ######################
    def _maxActuatorSteps(self):
        return torch.tensor(100000, dtype=self._dtype, device=self._device)

    # override
    def alignmentFromActuatorSteps(self, actuator_steps: torch.Tensor, env_state: typing.Optional[typing.Dict[str,torch.Tensor]] = None) -> torch.Tensor :
        self.predictDisturbances(actuator_steps=actuator_steps, env_state=env_state)

        cosys = COORDS.translation4x4(trans_vec=self._position(), dtype=self._dtype, device=self._device)
        cosys = cosys @ self._concentrator_cosys.cosysFromAngle()

        normal = self._normalFromCosys(cosys=cosys)
        pivoting_point = self._pivotingPointFromCosys(cosys=cosys)
        side_east = self._sideEastFromCosys(cosys=cosys)
        side_up = self._sideUpFromCosys(cosys=cosys)
        return normal, pivoting_point, side_east, side_up, actuator_steps, cosys


class TwoAxesAlignmentModel(AbstractAlignmentModelWithDisturbanceModel):
    class Keys(typing.NamedTuple):
        position_east : str                         = 'position_east'
        position_north : str                        = 'position_north'
        position_up : str                           = 'position_up'

        # actuator 1 keys
        actuator_1_increment : str                  = 'actuator_1_increment'
        actuator_1_initial_angle : str              = 'actuator_1_initial_angle'
        actuator_1_actuator_offset : str            = 'actuator_1_actuator_offset'
        actuator_1_joint_radius : str               = 'actuator_1_joint_radius'
        actuator_1_initial_stroke : str             = 'actuator_1_initial_stroke'
        actuator_1_min_steps : str                  = 'actuator_1_min_steps'
        actuator_1_max_steps : str                  = 'actuator_1_max_steps'
        actuator_1_rotating_clockwise : str         = 'actuator_1_rotating_clockwise'

        # actuator 2 keys
        actuator_2_increment : str                  = 'actuator_2_increment'
        actuator_2_initial_angle : str              = 'actuator_2_initial_angle'
        actuator_2_actuator_offset : str            = 'actuator_2_actuator_offset'
        actuator_2_joint_radius : str               = 'actuator_2_joint_radius'
        actuator_2_initial_stroke : str             = 'actuator_2_initial_stroke'
        actuator_2_min_steps : str                  = 'actuator_2_max_steps'
        actuator_2_max_steps : str                  = 'actuator_2_max_steps'
        actuator_2_min_steps : str                  = 'actuator_2_min_steps'
        actuator_2_max_steps : str                  = 'actuator_2_max_steps'
        actuator_2_rotating_clockwise : str         = 'actuator_2_rotating_clockwise'

        # joint 1 keys
        joint_1_cosys_pivot_east : str              = 'joint_1_pivot_east'
        joint_1_cosys_pivot_north : str             = 'joint_1_pivot_north'
        joint_1_cosys_pivot_up : str                = 'joint_1_pivot_up'
        joint_1_east_tilt : str                     = 'joint_1_east_tilt'
        joint_1_north_tilt : str                    = 'joint_1_north_tilt'
        joint_1_up_tilt : str                       = 'joint_1_up_tilt'

        # joint 2 keys
        joint_2_cosys_pivot_east : str              = 'joint_2_pivot_east'
        joint_2_cosys_pivot_north : str             = 'joint_2_pivot_north'
        joint_2_cosys_pivot_up : str                = 'joint_2_pivot_up'
        joint_2_east_tilt : str                     = 'joint_2_east_tilt'
        joint_2_north_tilt : str                    = 'joint_2_north_tilt'
        joint_2_up_tilt : str                       = 'joint_2_up_tilt'

        # concentrator keys
        concentrator_cosys_pivot_east : str          = 'concentrator_pivot_east'
        concentrator_cosys_pivot_north : str         = 'concentrator_pivot_north'
        concentrator_cosys_pivot_up : str            = 'concentrator_pivot_up'
        concentrator_east_tilt : str                 = 'concentrator_east_tilt'
        concentrator_north_tilt : str                = 'concentrator_north_tilt'
        concentrator_up_tilt : str                   = 'concentrator_up_tilt'
    
    keys = Keys()

    class DistKeys(typing.NamedTuple):
        position_azim : str                    = 'position_azim'
        position_elev : str                    = 'position_elev'
        position_rad : str                     = 'position_rad'

        # actuator 1 keys
        actuator_1_increment : str                  = 'actuator_1_increment'
        actuator_1_initial_angle : str              = 'actuator_1_initial_angle'
        actuator_1_actuator_offset : str            = 'actuator_1_actuator_offset'
        actuator_1_joint_radius : str               = 'actuator_1_joint_radius'
        actuator_1_initial_stroke : str             = 'actuator_1_initial_stroke'

        # actuator 2 keys
        actuator_2_increment : str                  = 'actuator_2_increment'
        actuator_2_initial_angle : str              = 'actuator_2_initial_angle'
        actuator_2_actuator_offset : str            = 'actuator_2_actuator_offset'
        actuator_2_joint_radius : str               = 'actuator_2_joint_radius'
        actuator_2_initial_stroke : str             = 'actuator_2_initial_stroke'

        # joint 1 keys
        joint_1_cosys_pivot_azim : str         = 'joint_1_cosys_pivot_azim'
        joint_1_cosys_pivot_elev : str         = 'joint_1_cosys_pivot_elev'
        joint_1_cosys_pivot_rad : str          = 'joint_1_cosys_pivot_rad'
        joint_1_east_tilt : str                = 'joint_1_east_tilt'
        joint_1_north_tilt : str               = 'joint_1_north_tilt'
        joint_1_up_tilt : str                  = 'joint_1_up_tilt'

        # joint 2 keys
        joint_2_cosys_pivot_azim : str         = 'joint_2_cosys_pivot_azim'
        joint_2_cosys_pivot_elev : str         = 'joint_2_cosys_pivot_elev'
        joint_2_cosys_pivot_rad : str          = 'joint_2_cosys_pivot_rad'
        joint_2_east_tilt : str                = 'joint_2_east_tilt'
        joint_2_north_tilt : str               = 'joint_2_north_tilt'
        joint_2_up_tilt : str                  = 'joint_2_up_tilt'

        # concentrator keys
        concentrator_cosys_pivot_azim : str     = 'concentrator_cosys_pivot_azim'
        concentrator_cosys_pivot_elev : str     = 'concentrator_cosys_pivot_elev'
        concentrator_cosys_pivot_rad : str      = 'concentrator_cosys_pivot_rad'
        concentrator_east_tilt : str            = 'concentrator_east_tilt'
        concentrator_north_tilt : str           = 'concentrator_north_tilt'
        concentrator_up_tilt : str              = 'concentrator_up_tilt'

    dist_keys = DistKeys()

    def __init__(self,
                 # parametrization
                 parameter_dict : typing.Dict[str, torch.Tensor],
                 
                 # first axis
                 actuator_1 : Actuators.AbstractActuator,
                 joint_1 : Joints.AbstractJoint,
                 concentrator_cosys : Joints.FixedJoint,

                 # second axis
                 actuator_2 : typing.Optional[Actuators.AbstractActuator] = None,
                 joint_2 : typing.Optional[Joints.AbstractJoint] = None,

                 # parameter keys
                 position_east_key : typing.Optional[str] = None,
                 position_north_key : typing.Optional[str] = None,
                 position_up_key : typing.Optional[str] = None,

                 # disturbance keys
                 position_azim_dist_key : typing.Optional[str] = None,
                 position_elev_dist_key : typing.Optional[str] = None,
                 position_rad_dist_key : typing.Optional[str] = None,

                 # disturbance model
                 disturbance_model : typing.Optional[DM.AbstractAlignmentDisturbanceModel] = None,

                 # disturbance factors
                 dist_factor_rot: typing.Optional[torch.Tensor] = None,
                 dist_factor_len: typing.Optional[torch.Tensor] = None,

                 # approximation factors:
                 max_num_epochs: int = 20,
                 eps: torch.Tensor = torch.tensor(10**(-12)),

                 # pytorch config
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.get_default_dtype(),
                ):

        super().__init__(
                         parameter_dict=parameter_dict,
                         disturbance_model=disturbance_model,
                         position_east_key=position_east_key,
                         position_north_key=position_north_key,
                         position_up_key=position_up_key,
                         position_azim_dist_key=position_azim_dist_key,
                         position_elev_dist_key=position_elev_dist_key,
                         position_rad_dist_key=position_rad_dist_key,
                         dist_factor_rot=dist_factor_rot,
                         dist_factor_len=dist_factor_len,
                         dtype=dtype,
                         device=device
                         )

        self.keys = TwoAxesAlignmentModel.Keys()
        self.dist_keys = TwoAxesAlignmentModel.DistKeys()

        self._actuator_1 = actuator_1
        self._actuator_2 = actuator_2
        self._joint_1 = joint_1
        self._joint_2 = joint_2
        self._concentrator_cosys = concentrator_cosys

        self._max_num_epochs = max_num_epochs
        self._eps = eps

    # override
    def _setDisturbances(self, disturbance_dict: typing.Dict[str, torch.Tensor]):
        super()._setDisturbances(disturbance_dict=disturbance_dict)

        self._actuator_1.setDisturbanceDict(disturbance_dict=disturbance_dict)
        self._actuator_2.setDisturbanceDict(disturbance_dict=disturbance_dict)
        self._joint_1.setDisturbanceDict(disturbance_dict=disturbance_dict)
        self._joint_2.setDisturbanceDict(disturbance_dict=disturbance_dict)
        self._concentrator_cosys.setDisturbanceDict(disturbance_dict=disturbance_dict)

    # override
    def fixateDisturbances(self) -> typing.Dict[str, torch.Tensor]:
        disturbance_dict = super().fixateDisturbances()
        if self.dist_keys.position_rad in disturbance_dict:
            pos_offset = self._parametrizedVector(parameter_keys = [None], disturbance_keys=[self.dist_keys.position_azim, self.dist_keys.position_elev, self.dist_keys.position_rad])
            self._parameter_dict[self.keys.position_east] += pos_offset[COORDS.E]
            self._parameter_dict[self.keys.position_north] += pos_offset[COORDS.N]
            self._parameter_dict[self.keys.position_up] += pos_offset[COORDS.U]

        if self.dist_keys.concentrator_cosys_pivot_rad in disturbance_dict:
            conc_offset = self._parametrizedVector(parameter_keys = [None], disturbance_keys=[self.dist_keys.concentrator_cosys_pivot_azim, self.dist_keys.concentrator_cosys_pivot_elev, self.dist_keys.concentrator_cosys_pivot_rad])
            self._parameter_dict[self.keys.concentrator_cosys_pivot_east] += conc_offset[COORDS.E]
            self._parameter_dict[self.keys.concentrator_cosys_pivot_north] += conc_offset[COORDS.N]
            self._parameter_dict[self.keys.concentrator_cosys_pivot_up] += conc_offset[COORDS.U]

        if self.dist_keys.joint_1_cosys_pivot_rad in disturbance_dict:
            j1_offset = self._parametrizedVector(parameter_keys = [None], disturbance_keys=[self.dist_keys.joint_1_cosys_pivot_azim, self.dist_keys.joint_1_cosys_pivot_elev, self.dist_keys.joint_1_cosys_pivot_rad])
            self._parameter_dict[self.keys.joint_1_cosys_pivot_east] += j1_offset[COORDS.E]
            self._parameter_dict[self.keys.joint_1_cosys_pivot_north] += j1_offset[COORDS.N]
            self._parameter_dict[self.keys.joint_1_cosys_pivot_up] += j1_offset[COORDS.U]

        if self.dist_keys.joint_2_cosys_pivot_rad in disturbance_dict:
            j2_offset = self._parametrizedVector(parameter_keys = [None], disturbance_keys=[self.dist_keys.joint_2_cosys_pivot_azim, self.dist_keys.joint_2_cosys_pivot_elev, self.dist_keys.joint_2_cosys_pivot_rad])
            self._parameter_dict[self.keys.joint_2_cosys_pivot_east] += j2_offset[COORDS.E]
            self._parameter_dict[self.keys.joint_2_cosys_pivot_north] += j2_offset[COORDS.N]
            self._parameter_dict[self.keys.joint_2_cosys_pivot_up] += j2_offset[COORDS.U]

    ######################
    #-   Normalization   -#
    ######################
    def _maxActuatorSteps(self):
        if self._actuator_1._max_actuator_steps() >= self._actuator_2._max_actuator_steps():
            return self._actuator_1._max_actuator_steps()
        else:
            return self._actuator_2._max_actuator_steps()

    ############################
    #-   Forward Kinematics   -#
    ############################
    def checkForNan(self) -> bool:
        for tp in self._disturbance_dict.values():
            if torch.isnan(tp):
                return True
        return False

    # override
    def alignmentFromActuatorSteps(self, actuator_steps: torch.Tensor, env_state: typing.Optional[typing.Dict[str,torch.Tensor]] = None) -> torch.Tensor :
        self.predictDisturbances(actuator_steps=actuator_steps, env_state=env_state)

        actuator_1_steps = actuator_steps[0]
        actuator_2_steps = actuator_steps[1]

        cosys = self._concentratorCosysFromSteps(actuator_1_steps=actuator_1_steps, actuator_2_steps=actuator_2_steps)
        normal = self._normalFromCosys(cosys=cosys)
        pivoting_point = self._pivotingPointFromCosys(cosys=cosys)
        side_east = self._sideEastFromCosys(cosys=cosys)
        side_up = self._sideUpFromCosys(cosys=cosys)
        return normal, pivoting_point, side_east, side_up, actuator_steps, cosys

    def _joint1CosysFromSteps(self, actuator_1_steps: typing.Optional[torch.Tensor] = None) -> torch.Tensor :
        if not actuator_1_steps:
            actuator_1_steps = torch.tensor(0, dtype = self._dtype, device=self._device, requires_grad=False)

        joint_1_angle = self._actuator_1.angleFromSteps(actuator_steps=actuator_1_steps)
        # joint_1_angle_deg = torch.rad2deg(joint_1_angle) # for debugging

        cosys = COORDS.translation4x4(trans_vec=self._position(), dtype=self._dtype, device=self._device)
        cosys = cosys @ self._joint_1.cosysFromAngle(angle=joint_1_angle)
        return cosys

    def _joint1CosysFromAngle(self, joint_1_angle: typing.Optional[torch.Tensor] = None) -> torch.Tensor :
        if not joint_1_angle:
            joint_1_angle = torch.tensor(0, dtype = self._dtype, device=self._device, requires_grad=False)

        cosys = COORDS.translation4x4(trans_vec=self._position(), dtype=self._dtype, device=self._device)
        cosys = cosys @ self._joint_1.cosysFromAngle(angle=joint_1_angle)
        return cosys

    def _joint2CosysFromSteps(self, 
                              actuator_1_steps: typing.Optional[torch.Tensor] = None,
                              actuator_2_steps: typing.Optional[torch.Tensor] = None
                              ) -> torch.Tensor :
        if self._joint_2:
            if not actuator_2_steps:
                actuator_2_steps = torch.tensor(0, dtype = self._dtype, device=self._device, requires_grad=False)

            joint_2_angle = self._actuator_2.angleFromSteps(actuator_steps=actuator_2_steps)
            # joint_2_angle_deg = torch.rad2deg(joint_2_angle) # for debugging
            
            cosys = self._joint1CosysFromSteps(actuator_1_steps=actuator_1_steps)
            cosys = cosys @ self._joint_2.cosysFromAngle(angle=joint_2_angle)
        else:
            cosys = COORDS.identity4x4(device=self._device)
        return cosys

    def _joint2CosysFromAngle(self, 
                              joint_1_angle: typing.Optional[torch.Tensor] = None,
                              joint_2_angle: typing.Optional[torch.Tensor] = None
                              ) -> torch.Tensor :

        if self._joint_2:
            if not joint_2_angle:
                joint_2_angle = torch.tensor(0, dtype = self._dtype, device=self._device, requires_grad=False)

            cosys = self._joint1CosysFromAngle(joint_1_angle=joint_1_angle)
            cosys = cosys @ self._joint_2.cosysFromAngle(angle=joint_2_angle)
        else:
            cosys = COORDS.identity4x4(dtype=self._dtype, device=self._device)
        return cosys

    def _concentratorCosysFromSteps(self, 
                              actuator_1_steps: typing.Optional[torch.Tensor] = None,
                              actuator_2_steps: typing.Optional[torch.Tensor] = None
                              ) -> torch.Tensor :
        cosys = self._joint2CosysFromSteps(actuator_1_steps=actuator_1_steps, actuator_2_steps=actuator_2_steps)
        cosys = cosys @ self._concentrator_cosys.cosysFromAngle()
        return cosys

    def _concentratorCosysFromAngle(self, 
                              joint_1_angle: typing.Optional[torch.Tensor] = None,
                              joint_2_angle: typing.Optional[torch.Tensor] = None
                              ) -> torch.Tensor :
        cosys = self._joint2CosysFromAngle(joint_1_angle=joint_1_angle, joint_2_angle=joint_2_angle)
        cosys = cosys @ self._concentrator_cosys.cosysFromAngle()
        return cosys

    ############################
    #-   Inverse Kinematics   -#
    ############################
    # override
    def alignmentFromSourceVec(self, 
                               to_source: torch.Tensor, 
                               aimpoint: torch.Tensor,
                               env_state: typing.Optional[typing.Dict[str,torch.Tensor]] = None,
                               ) -> typing.List[torch.Tensor]:
        actuator_steps = torch.tensor([0.0, 0.0], dtype=self._dtype, device=self._device, requires_grad=True)

        # loop criteria
        loss_history = []
        cosys = None
        epoch = 0
        side_east = None
        side_up = None

        # training loop
        while epoch < self._max_num_epochs:
            self.predictDisturbances(actuator_steps=actuator_steps, env_state=env_state)

            normal, pivoting_point, side_east, side_up, actuator_steps, cosys = self.alignmentFromActuatorSteps(actuator_steps=actuator_steps)

            # optimal normal for current orientation
            target_reflect_vec = aimpoint - pivoting_point
            target_reflect_vec = target_reflect_vec / target_reflect_vec.norm()
            target_normal = to_source + target_reflect_vec
            target_normal = target_normal / target_normal.norm()

            # loss
            mse_loss = torch.nn.MSELoss()
            loss_history.append(mse_loss(normal, target_normal))

            # step: retreive axis angles related to optimal normal for current orientation
            actuator_steps = self._actuatorStepsFromNormal(normal = target_normal, actuator_steps_guess=actuator_steps)

            # escape condition
            if epoch > 0:
                if torch.abs(loss_history[epoch] - loss_history[epoch-1]) <= self._eps:
                    # print('eps_break')
                    break
            
            # continue loop
            epoch = epoch + 1

        if epoch >= self._max_num_epochs:
            print('WARNING: kinematic maxed out approx. iterations!')

        side_east = self._sideEastFromCosys(cosys=cosys)
        side_up = self._sideUpFromCosys(cosys=cosys)
        return normal, pivoting_point, side_east, side_up, actuator_steps, cosys

    # override
    def alignmentFromSourcePos(self, 
                               source_pos: torch.Tensor,
                               aimpoint: torch.Tensor,
                               env_state: typing.Optional[typing.Dict[str,torch.Tensor]] = None,
                               ) -> typing.List[torch.Tensor]:
        actuator_steps = torch.tensor([0.0, 0.0], dtype=self._dtype, device=self._device, requires_grad=True)
        

        # loop criteria
        loss_history = []
        cosys = None
        epoch = 0

        side_east = None
        side_up = None

        # training loop
        while epoch < self._max_num_epochs:
            self.predictDisturbances(actuator_steps=actuator_steps, env_state=env_state)

            normal, pivoting_point, side_east, side_up, actuator_steps, cosys = self.alignmentFromActuatorSteps(actuator_steps=actuator_steps)

            # optimal normal for current orientation
            target_reflect_vec = aimpoint - pivoting_point
            target_reflect_vec = target_reflect_vec / target_reflect_vec.norm()
            to_source = source_pos - pivoting_point
            to_source = to_source / to_source.norm()
            target_normal = to_source + target_reflect_vec
            target_normal = target_normal / target_normal.norm()

            # loss
            mse_loss = torch.nn.MSELoss()
            loss_history.append(mse_loss(normal, target_normal))

            # step: retreive axis angles related to optimal normal for current orientation
            actuator_steps = self._actuatorStepsFromNormal(normal = target_normal, actuator_steps_guess=actuator_steps)

            # escape condition
            if epoch > 0:
                if torch.abs(loss_history[epoch] - loss_history[epoch-1]) <= self._eps:
                    # print('eps_break')
                    break
            
            # continue loop
            epoch = epoch + 1

        if epoch >= self._max_num_epochs:
            print('WARNING: kinematic maxed out approx. iterations!')

        return normal, pivoting_point, side_east, side_up, actuator_steps, cosys

    def _actuatorStepsFromJointAngles(self, joint_angles: torch.Tensor) -> torch.Tensor:
        
        actuator_1_steps = self._actuator_1.stepsFromAngle(angle=joint_angles[0])
        actuator_2_steps = self._actuator_2.stepsFromAngle(angle=joint_angles[1])
        actuator_steps = torch.tensor([actuator_1_steps, actuator_2_steps], dtype=self._dtype, device=self._device)
        
        return actuator_steps
        
    def _actuatorStepsFromAlignment(self, cosys: torch.Tensor) -> torch.Tensor:
        normal = self._normalFromCosys(cosys=cosys)
        actuator_steps = self._actuatorStepsFromNormal(normal=normal)
        return actuator_steps

    def _actuatorStepsFromNormal(self, normal: torch.Tensor, actuator_steps_guess: typing.Optional[torch.Tensor] = None) -> torch.Tensor:
        
        actuator_steps_guess = actuator_steps_guess if isinstance(actuator_steps_guess, torch.Tensor) else torch.tensor([0,0], dtype=self._dtype, device=self._device)

        # rotate normal into joint 1 cosys
        normal4x1 = torch.hstack([normal, torch.zeros(1, dtype=self._dtype, device=self._device)])
        joint_1_cosys = self._joint1CosysFromAngle()
        normal_j1 = joint_1_cosys.T @ normal4x1

        if self._joint_1.rotationDirection() == Joints.AbstractJoint.east_rotation_direction \
            and self._joint_2.rotationDirection() == Joints.AbstractJoint.up_rotation_direction:
            joint_angles = self._jointAnglesFromNormal_analytic_east_up(normal_j1=normal_j1, actuator_steps_guess=actuator_steps_guess)
        else:
            raise Exception("Invalid axis combination")

        actuator_steps = self._actuatorStepsFromJointAngles(joint_angles=joint_angles)

        return actuator_steps

    def _jointAnglesFromNormal_analytic_east_up(self, normal_j1: torch.Tensor, actuator_steps_guess: torch.Tensor) -> torch.Tensor:
        sin_2e = torch.sin(self._joint_2._east_tilt())
        cos_2e = torch.cos(self._joint_2._east_tilt())

        sin_2n = torch.sin(self._joint_2._north_tilt())
        cos_2n = torch.cos(self._joint_2._north_tilt())
        
        # rot angle 2
        calc_step_1 = normal_j1[COORDS.E] / cos_2n
        joint_2_angle = torch.arcsin(calc_step_1)
        assert(not torch.isnan(joint_2_angle)), ' normal_j1[E]: ' + str(normal_j1[COORDS.E]) + ' cos_2n: ' + str(cos_2n)

        sin_2u = torch.sin(joint_2_angle)
        cos_2u = torch.cos(joint_2_angle)

        # rot angle 1
        a = - cos_2e * cos_2u + sin_2e * sin_2n * sin_2u
        b = - sin_2e * cos_2u - cos_2e * sin_2n * sin_2u

        numerator = a * normal_j1[COORDS.U] - b * normal_j1[COORDS.N]
        denominator = a * normal_j1[COORDS.N] + b * normal_j1[COORDS.U]

        joint_1_angle = torch.arctan2(numerator, denominator)
        assert(not torch.isnan(joint_1_angle)), 'numerator: ' + str(numerator) + '\ndenominator: ' + str(denominator) \
                                             + '\nsin_2e: ' + str(sin_2e) + '\ncos_2e: ' + str(cos_2e) \
                                             + '\nsin_2n: ' + str(sin_2n) + '\ncos_2n: ' + str(cos_2n) \
                                             + '\nsin_2u: ' + str(sin_2u) + '\ncos_2u: ' + str(cos_2u) \
        
        joint_1_angle = joint_1_angle #- self._joint_1._east_tilt()
        joint_2_angle = joint_2_angle - self._joint_2._up_tilt()

        joint_angles = torch.tensor([joint_1_angle, joint_2_angle], dtype=self._dtype, device=self._device)
        # print(torch.rad2deg(rot_angles))
        return joint_angles

class HeliokonAlignmentModel(TwoAxesAlignmentModel):

    def __init__(self,
                 # parametrization
                 disturbance_model : typing.Optional[DM.AbstractAlignmentDisturbanceModel] = None,
                 parameter_dict : typing.Optional[typing.Dict[str, torch.Tensor]]    = None,

                 position: typing.Optional[torch.Tensor]                             = None,
                 # actuator 1 parametrization
                 actuator_1_actuator_origin: typing.Optional[torch.Tensor]           = None,
                 actuator_1_working_point_origin: typing.Optional[torch.Tensor]     = None,
                 actuator_1_initial_stroke_length: typing.Optional[torch.Tensor]     = None,
                 actuator_1_increment: typing.Optional[torch.Tensor]                 = None,
                 actuator_1_min_steps: typing.Optional[torch.Tensor]                 = None,
                 actuator_1_max_steps: typing.Optional[torch.Tensor]                 = None,
                 actuator_1_angle_offset_deg: typing.Optional[torch.Tensor]          = None,
                 actuator_1_rotating_clockwise : typing.Optional[torch.Tensor]       = None,
                 # actuator 2 parametrization
                 actuator_2_actuator_origin: typing.Optional[torch.Tensor]           = None,
                 actuator_2_working_point_origin: typing.Optional[torch.Tensor]      = None,
                 actuator_2_initial_stroke_length: typing.Optional[torch.Tensor]     = None,
                 actuator_2_increment: typing.Optional[torch.Tensor]                 = None,
                 actuator_2_min_steps: typing.Optional[torch.Tensor]                 = None,
                 actuator_2_max_steps: typing.Optional[torch.Tensor]                 = None,
                 actuator_2_angle_offset_deg: typing.Optional[torch.Tensor]          = None,
                 actuator_2_rotating_clockwise : typing.Optional[torch.Tensor]       = None,
                 # joints and concentrator parametrization
                 joint_2_pivot_point: typing.Optional[torch.Tensor]                  = None,
                 concentrator_pivot_point: typing.Optional[torch.Tensor]             = None,

                 # disturbance factors
                 dist_factor_rot: typing.Optional[torch.Tensor]                     = None,
                 dist_factor_len: typing.Optional[torch.Tensor]                     = None,
                 dist_factor_perc: typing.Optional[torch.Tensor]                    = None,

                 # approximation factors
                 max_num_epochs: int = 20,
                 eps: torch.Tensor = torch.tensor(10**(-12)),

                 # pytorch config
                 device: torch.device = torch.device('cpu'),
                 dtype: torch.dtype = torch.get_default_dtype(),
                ):

        self._device = device
        self._dtype = dtype
        
        parameter_dict = parameter_dict if parameter_dict \
                                        else self._createParameterDict(
                                            position=position,
                                            actuator_1_actuator_origin=actuator_1_actuator_origin,
                                            actuator_1_working_point_origin=actuator_1_working_point_origin,
                                            actuator_1_initial_stroke_length=actuator_1_initial_stroke_length,
                                            actuator_1_increment=actuator_1_increment,
                                            actuator_1_min_steps=actuator_1_min_steps,
                                            actuator_1_max_steps=actuator_1_max_steps,
                                            actuator_1_angle_offset_deg=actuator_1_angle_offset_deg,
                                            actuator_1_rotating_clockwise=actuator_1_rotating_clockwise,
                                            actuator_2_actuator_origin=actuator_2_actuator_origin,
                                            actuator_2_working_point_origin=actuator_2_working_point_origin,
                                            actuator_2_initial_stroke_length=actuator_2_initial_stroke_length,
                                            actuator_2_increment=actuator_2_increment,
                                            actuator_2_min_steps=actuator_2_min_steps,
                                            actuator_2_max_steps=actuator_2_max_steps,
                                            actuator_2_angle_offset_deg=actuator_2_angle_offset_deg,
                                            actuator_2_rotating_clockwise=actuator_2_rotating_clockwise,
                                            joint_2_pivot_point=joint_2_pivot_point,
                                            concentrator_pivot_point=concentrator_pivot_point,
                                        )

        actuator_1 = Actuators.LinearActuator(min_actuator_steps_key=   HeliokonAlignmentModel.keys.actuator_1_min_steps,
                                              max_actuator_steps_key=   HeliokonAlignmentModel.keys.actuator_1_max_steps,
                                              rotating_clockwise_key=   HeliokonAlignmentModel.keys.actuator_1_rotating_clockwise,
                                              increment_key=            HeliokonAlignmentModel.keys.actuator_1_increment,
                                              increment_dist_key=       HeliokonAlignmentModel.dist_keys.actuator_1_increment,
                                              initial_angle_key=        HeliokonAlignmentModel.keys.actuator_1_initial_angle,
                                              initial_angle_dist_key=   HeliokonAlignmentModel.dist_keys.actuator_1_initial_angle,
                                              actuator_offset_key=      HeliokonAlignmentModel.keys.actuator_1_actuator_offset,
                                              actuator_offset_dist_key= HeliokonAlignmentModel.dist_keys.actuator_1_actuator_offset,
                                              joint_radius_key=         HeliokonAlignmentModel.keys.actuator_1_joint_radius,
                                              joint_radius_dist_key=    HeliokonAlignmentModel.dist_keys.actuator_1_joint_radius,
                                              initial_stroke_key=       HeliokonAlignmentModel.keys.actuator_1_initial_stroke,
                                              initial_stroke_dist_key=  HeliokonAlignmentModel.dist_keys.actuator_1_initial_stroke,

                                              parameter_dict=           parameter_dict,

                                            #   rotating_right_handed=    False,
                                            #   max_actuator_steps=       actuator_1_max_steps,

                                              dist_factor_perc=         dist_factor_perc,
                                              dist_factor_rot=          dist_factor_rot,

                                              device=                   self._device,
                                              dtype=                    self._dtype
        )

        actuator_2 = Actuators.LinearActuator(min_actuator_steps_key=   HeliokonAlignmentModel.keys.actuator_2_min_steps,
                                              max_actuator_steps_key=   HeliokonAlignmentModel.keys.actuator_2_max_steps,
                                              rotating_clockwise_key=   HeliokonAlignmentModel.keys.actuator_2_rotating_clockwise,
                                              increment_key=            HeliokonAlignmentModel.keys.actuator_2_increment,
                                              increment_dist_key=       HeliokonAlignmentModel.dist_keys.actuator_2_increment,
                                              initial_angle_key=        HeliokonAlignmentModel.keys.actuator_2_initial_angle,
                                              initial_angle_dist_key=   HeliokonAlignmentModel.dist_keys.actuator_2_initial_angle,
                                              actuator_offset_key=      HeliokonAlignmentModel.keys.actuator_2_actuator_offset,
                                              actuator_offset_dist_key= HeliokonAlignmentModel.dist_keys.actuator_2_actuator_offset,
                                              joint_radius_key=         HeliokonAlignmentModel.keys.actuator_2_joint_radius,
                                              joint_radius_dist_key=    HeliokonAlignmentModel.dist_keys.actuator_2_joint_radius,
                                              initial_stroke_key=       HeliokonAlignmentModel.keys.actuator_2_initial_stroke,
                                              initial_stroke_dist_key=  HeliokonAlignmentModel.dist_keys.actuator_2_initial_stroke,

                                              parameter_dict=           parameter_dict,

                                            #   max_actuator_steps=       actuator_2_max_steps,

                                              dist_factor_perc=         dist_factor_perc,
                                              dist_factor_rot=          dist_factor_rot,

                                              device=                   self._device,
                                              dtype=                    self._dtype
        )

        joint_1 = Joints.EastRotationJoint(cosys_pivot_east_key=        HeliokonAlignmentModel.keys.joint_1_cosys_pivot_east,
                                           cosys_pivot_north_key=       HeliokonAlignmentModel.keys.joint_1_cosys_pivot_north,
                                           cosys_pivot_up_key=          HeliokonAlignmentModel.keys.joint_1_cosys_pivot_up,
                                           east_tilt_key=               HeliokonAlignmentModel.keys.joint_1_east_tilt,
                                           north_tilt_key=              HeliokonAlignmentModel.keys.joint_1_north_tilt,
                                           up_tilt_key=                 HeliokonAlignmentModel.keys.joint_1_up_tilt,
                                           cosys_pivot_azim_dist_key=   HeliokonAlignmentModel.dist_keys.joint_1_cosys_pivot_azim,
                                           cosys_pivot_elev_dist_key=   HeliokonAlignmentModel.dist_keys.joint_1_cosys_pivot_elev,
                                           cosys_pivot_rad_dist_key=    HeliokonAlignmentModel.dist_keys.joint_1_cosys_pivot_rad,
                                           east_tilt_dist_key=          HeliokonAlignmentModel.dist_keys.joint_1_east_tilt,
                                           north_tilt_dist_key=         HeliokonAlignmentModel.dist_keys.joint_1_north_tilt,
                                           up_tilt_dist_key=            HeliokonAlignmentModel.dist_keys.joint_1_up_tilt,

                                           parameter_dict=              parameter_dict,
                                           dist_factor_rot=             dist_factor_rot,
                                           dist_factor_len=             dist_factor_len,

                                           device=                      self._device,
                                           dtype =                      self._dtype
        )

        joint_2 = Joints.UpRotationJoint(  cosys_pivot_east_key=        HeliokonAlignmentModel.keys.joint_2_cosys_pivot_east,
                                           cosys_pivot_north_key=       HeliokonAlignmentModel.keys.joint_2_cosys_pivot_north,
                                           cosys_pivot_up_key=          HeliokonAlignmentModel.keys.joint_2_cosys_pivot_up,
                                           east_tilt_key=               HeliokonAlignmentModel.keys.joint_2_east_tilt,
                                           north_tilt_key=              HeliokonAlignmentModel.keys.joint_2_north_tilt,
                                           up_tilt_key=                 HeliokonAlignmentModel.keys.joint_2_up_tilt,
                                           cosys_pivot_azim_dist_key=   HeliokonAlignmentModel.dist_keys.joint_2_cosys_pivot_azim,
                                           cosys_pivot_elev_dist_key=   HeliokonAlignmentModel.dist_keys.joint_2_cosys_pivot_elev,
                                           cosys_pivot_rad_dist_key=    HeliokonAlignmentModel.dist_keys.joint_2_cosys_pivot_rad,
                                           east_tilt_dist_key=          HeliokonAlignmentModel.dist_keys.joint_2_east_tilt,
                                           north_tilt_dist_key=         HeliokonAlignmentModel.dist_keys.joint_2_north_tilt,
                                           up_tilt_dist_key=            HeliokonAlignmentModel.dist_keys.joint_2_up_tilt,

                                           parameter_dict=              parameter_dict,
                                           dist_factor_rot=             dist_factor_rot,
                                           dist_factor_len=             dist_factor_len,

                                           device=                      self._device,
                                           dtype =                      self._dtype
        )

        concentrator_cosys = Joints.FixedJoint( cosys_pivot_east_key=        HeliokonAlignmentModel.keys.concentrator_cosys_pivot_east,
                                                cosys_pivot_north_key=       HeliokonAlignmentModel.keys.concentrator_cosys_pivot_north,
                                                cosys_pivot_up_key=          HeliokonAlignmentModel.keys.concentrator_cosys_pivot_up,
                                                east_tilt_key=               HeliokonAlignmentModel.keys.concentrator_east_tilt,
                                                north_tilt_key=              HeliokonAlignmentModel.keys.concentrator_north_tilt,
                                                up_tilt_key=                 HeliokonAlignmentModel.keys.concentrator_up_tilt,
                                                cosys_pivot_azim_dist_key=   HeliokonAlignmentModel.dist_keys.concentrator_cosys_pivot_azim,
                                                cosys_pivot_elev_dist_key=   HeliokonAlignmentModel.dist_keys.concentrator_cosys_pivot_elev,
                                                cosys_pivot_rad_dist_key=    HeliokonAlignmentModel.dist_keys.concentrator_cosys_pivot_rad,
                                                east_tilt_dist_key=          HeliokonAlignmentModel.dist_keys.concentrator_east_tilt,
                                                north_tilt_dist_key=         HeliokonAlignmentModel.dist_keys.concentrator_north_tilt,
                                                up_tilt_dist_key=            HeliokonAlignmentModel.dist_keys.concentrator_up_tilt,

                                                parameter_dict=              parameter_dict,
                                                dist_factor_rot=             dist_factor_rot,
                                                dist_factor_len=             dist_factor_len,

                                                device=                      self._device,
                                                dtype =                      self._dtype
        )

        super().__init__(parameter_dict=parameter_dict,
                         disturbance_model=disturbance_model,
                         actuator_1=actuator_1,
                         actuator_2=actuator_2,
                         joint_1=joint_1,
                         joint_2=joint_2,
                         concentrator_cosys=concentrator_cosys,
                         dist_factor_rot=dist_factor_rot,
                         dist_factor_len=dist_factor_len,
                         max_num_epochs=max_num_epochs,
                         eps=eps,
                         device=device,
                         dtype=dtype
                        )
    
    # override
    def summary(self) -> str:
        summary_str = super().summary()
        summary_str = summary_str + "Heliokon\n\n"
        if self._disturbance_model:
            summary_str = summary_str + self._disturbance_model.summary()
        else:
            summary_str = summary_str + "No disturbance model.\n"
        return summary_str

    def _createParameterDict(self, 
                                position: typing.Optional[torch.Tensor]                             = None,
                                # actuator 1 parametrization
                                actuator_1_actuator_origin: typing.Optional[torch.Tensor]           = None,
                                actuator_1_working_point_origin: typing.Optional[torch.Tensor]     = None,
                                actuator_1_initial_stroke_length: typing.Optional[torch.Tensor]     = None,
                                actuator_1_increment: typing.Optional[torch.Tensor]                 = None,
                                actuator_1_min_steps: typing.Optional[torch.Tensor]                 = None,
                                actuator_1_max_steps: typing.Optional[torch.Tensor]                 = None,
                                actuator_1_angle_offset_deg: typing.Optional[torch.Tensor]          = None,
                                actuator_1_rotating_clockwise: typing.Optional[bool]                = None,
                                # actuator 2 parametrization
                                actuator_2_actuator_origin: typing.Optional[torch.Tensor]           = None,
                                actuator_2_working_point_origin: typing.Optional[torch.Tensor]      = None,
                                actuator_2_initial_stroke_length: typing.Optional[torch.Tensor]     = None,
                                actuator_2_increment: typing.Optional[torch.Tensor]                 = None,
                                actuator_2_min_steps: typing.Optional[torch.Tensor]                 = None,
                                actuator_2_max_steps: typing.Optional[torch.Tensor]                 = None,
                                actuator_2_angle_offset_deg: typing.Optional[torch.Tensor]          = None,
                                actuator_2_rotating_clockwise: typing.Optional[bool]                = None,
                                # joints and concentrator parametrization
                                joint_2_pivot_point: typing.Optional[torch.Tensor]                  = None,
                                concentrator_pivot_point: typing.Optional[torch.Tensor]             = None,
                                ):

        position                            = position                          if isinstance(position, torch.Tensor) else torch.tensor([0, 0, 0],                dtype=self._dtype, device=self._device)

        actuator_1_actuator_origin          = actuator_1_actuator_origin        if actuator_1_actuator_origin       else torch.tensor([0, 0.294, -0.172],       dtype=self._dtype, device=self._device)
        actuator_1_working_point_origin     = actuator_1_working_point_origin   if actuator_1_working_point_origin  else torch.tensor([0, 0.0585, 0.315],       dtype=self._dtype, device=self._device)
        actuator_1_initial_stroke_length    = actuator_1_initial_stroke_length  if actuator_1_initial_stroke_length else torch.tensor(0.075,                    dtype=self._dtype, device=self._device)
        actuator_1_increment                = actuator_1_increment              if actuator_1_increment             else torch.tensor(154166.666,               dtype=self._dtype, device=self._device)
        actuator_1_min_steps                = actuator_1_min_steps              if actuator_1_min_steps             else torch.tensor(0,                        dtype=self._dtype, device=self._device)
        actuator_1_max_steps                = actuator_1_max_steps              if actuator_1_max_steps             else torch.tensor(69525,                    dtype=self._dtype, device=self._device)
        actuator_1_angle_offset_deg         = actuator_1_angle_offset_deg       if actuator_1_angle_offset_deg      else torch.tensor(90,                       dtype=self._dtype, device=self._device)
        actuator_1_rotating_clockwise       = actuator_1_rotating_clockwise     if actuator_1_rotating_clockwise    else True

        actuator_2_actuator_origin          = actuator_2_actuator_origin        if actuator_2_actuator_origin       else torch.tensor([0.11, 0.33, 0],          dtype=self._dtype, device=self._device)
        actuator_2_working_point_origin     = actuator_2_working_point_origin   if actuator_2_working_point_origin  else torch.tensor([0.309, 0, 0],            dtype=self._dtype, device=self._device)
        actuator_2_initial_stroke_length    = actuator_2_initial_stroke_length  if actuator_2_initial_stroke_length else torch.tensor(0.075,                    dtype=self._dtype, device=self._device)
        actuator_2_increment                = actuator_2_increment              if actuator_2_increment             else torch.tensor(154166.666,               dtype=self._dtype, device=self._device)
        actuator_2_min_steps                = actuator_2_min_steps              if actuator_2_min_steps             else torch.tensor(0,                        dtype=self._dtype, device=self._device)
        actuator_2_max_steps                = actuator_2_max_steps              if actuator_2_max_steps             else torch.tensor(75690,                    dtype=self._dtype, device=self._device)
        actuator_2_angle_offset_deg         = actuator_2_angle_offset_deg       if actuator_2_angle_offset_deg      else torch.tensor(55,                       dtype=self._dtype, device=self._device)
        actuator_2_rotating_clockwise       = actuator_2_rotating_clockwise     if actuator_2_rotating_clockwise    else False
        
        joint_2_pivot_point                 = joint_2_pivot_point               if joint_2_pivot_point              else torch.tensor([0, 0, 0.315],            dtype=self._dtype, device=self._device)
        concentrator_pivot_point            = concentrator_pivot_point          if concentrator_pivot_point         else torch.tensor([0, -0.17755, -0.4045],   dtype=self._dtype, device=self._device)

        actuator_1_joint_origin = torch.tensor([0,0,0], dtype=self._dtype, device=self._device, requires_grad=False)
        actuator_1_actuator_offset, actuator_1_joint_radius = Actuators.LinearActuator.parametersFromPositions(actuator_origin=actuator_1_actuator_origin,
                                                                                                                initial_working_point=actuator_1_working_point_origin,
                                                                                                                joint_origin=actuator_1_joint_origin,
                                                                                                                )

        actuator_2_joint_origin = torch.tensor([0,0,0], dtype=self._dtype, device=self._device, requires_grad=False)
        actuator_2_actuator_offset, actuator_2_joint_radius = Actuators.LinearActuator.parametersFromPositions(actuator_origin=actuator_2_actuator_origin,
                                                                                                                initial_working_point=actuator_2_working_point_origin,
                                                                                                                joint_origin=actuator_2_joint_origin,
                                                                                                                )        

        # create parameter dict
        parameter_dict = {
                            # position
                            HeliokonAlignmentModel.keys.position_east                            : position[COORDS.E],
                            HeliokonAlignmentModel.keys.position_north                           : position[COORDS.N],
                            HeliokonAlignmentModel.keys.position_up                              : position[COORDS.U],

                            # actuator 1
                            HeliokonAlignmentModel.keys.actuator_1_increment                     : actuator_1_increment,
                            HeliokonAlignmentModel.keys.actuator_1_actuator_offset               : actuator_1_actuator_offset,
                            HeliokonAlignmentModel.keys.actuator_1_joint_radius                  : actuator_1_joint_radius,
                            HeliokonAlignmentModel.keys.actuator_1_initial_stroke                : actuator_1_initial_stroke_length,
                            HeliokonAlignmentModel.keys.actuator_1_initial_angle                 : torch.deg2rad(actuator_1_angle_offset_deg),
                            HeliokonAlignmentModel.keys.actuator_1_min_steps                     : actuator_1_min_steps,
                            HeliokonAlignmentModel.keys.actuator_1_max_steps                     : actuator_1_max_steps,
                            HeliokonAlignmentModel.keys.actuator_1_rotating_clockwise            : actuator_1_rotating_clockwise,

                            # actuator 2
                            HeliokonAlignmentModel.keys.actuator_2_increment                     : actuator_2_increment,
                            HeliokonAlignmentModel.keys.actuator_2_actuator_offset               : actuator_2_actuator_offset,
                            HeliokonAlignmentModel.keys.actuator_2_joint_radius                  : actuator_2_joint_radius,
                            HeliokonAlignmentModel.keys.actuator_2_initial_stroke                : actuator_2_initial_stroke_length,
                            HeliokonAlignmentModel.keys.actuator_2_initial_angle                 : torch.deg2rad(actuator_2_angle_offset_deg),
                            HeliokonAlignmentModel.keys.actuator_2_min_steps                     : actuator_2_min_steps,
                            HeliokonAlignmentModel.keys.actuator_2_max_steps                     : actuator_2_max_steps,
                            HeliokonAlignmentModel.keys.actuator_2_rotating_clockwise            : actuator_2_rotating_clockwise,

                            # joint 2
                            HeliokonAlignmentModel.keys.joint_2_cosys_pivot_east                 : joint_2_pivot_point[COORDS.E],
                            HeliokonAlignmentModel.keys.joint_2_cosys_pivot_north                : joint_2_pivot_point[COORDS.N],
                            HeliokonAlignmentModel.keys.joint_2_cosys_pivot_up                   : joint_2_pivot_point[COORDS.U],

                            # concentrator
                            HeliokonAlignmentModel.keys.concentrator_cosys_pivot_east            : concentrator_pivot_point[COORDS.E],
                            HeliokonAlignmentModel.keys.concentrator_cosys_pivot_north           : concentrator_pivot_point[COORDS.N],
                            HeliokonAlignmentModel.keys.concentrator_cosys_pivot_up              : concentrator_pivot_point[COORDS.U],
        }

        return parameter_dict

# main
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    torch.set_default_dtype(torch.float64)
    dtype = torch.get_default_dtype()
    disturbance_list = list(HeliokonAlignmentModel.keys._fields)
    # disturbance_model = DM.RigidBodyAlignmentDisturbanceModel(disturbance_list=disturbance_list, randomize_initial_disturbances=False, initial_disturbance_range=5.0, dtype=dtype)
    disturbance_model = DM.SNNAlignmentDisturbanceModel(disturbance_list=disturbance_list,
                                                     hidden_dim = 2,
                                                     n_layers = 3,
                                                     num_inputs = 2,
    )
    heliostat_alignment = HeliokonAlignmentModel(disturbance_model=disturbance_model, dtype=dtype, max_num_epochs=200)
    fig=plt.figure()
    ax = plt.axes(projection='3d')
    ax.set_xlim((-1,1))
    ax.set_ylim((-1,1))
    ax.set_zlim((-1,1))
    for ax1 in range(0, 69525, 30000):
        for ax2 in range(0, 75690, 10000):

            actuator_steps = torch.tensor([ax1, ax2], dtype=dtype)
            normal, pivoting_point, side_east, side_up, actuator_steps, cosys = heliostat_alignment.alignmentFromActuatorSteps(actuator_steps=actuator_steps)
            numpy_normal = normal.detach().numpy()
            print('FWD: Ax1: ' + str(ax1) + ', Ax2: ' + str(ax2) + ' -> ' + str(numpy_normal))
            ax.plot3D([0, numpy_normal[0]], [0, numpy_normal[1]], [0, numpy_normal[2]])
            actuator_steps_numpy = heliostat_alignment._actuatorStepsFromNormal(normal=normal).detach().numpy()
            print('BWD: Ax1: ' + str(actuator_steps_numpy[0]) + ', Ax2: ' + str(actuator_steps_numpy[1]))
    
    plt.show()
    exit()