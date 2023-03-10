# system dependencies
import torch
import typing
import datetime
from yacs.config import CfgNode
import pickle
import sys
import os

# local dependencies
module_dir = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(module_dir)
from HausdorffMetric import HausdorffMetric as HAUSDORFF

lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(lib_dir)
import CoordinateSystemsLib.ExtendedCoordinates as COORDS

class HeliostatDataPoint:
    _datetime_format = "%Y-%m-%d %H:%M:%S"

    class EnvStateKeys(typing.NamedTuple):
        month : str = 'Month'
        elapsed_days : str = 'Elapsed Days'
        elapsed_start : str = 'Elapsed Start'
        elapsed_end : str = 'Elapsed End'
    env_keys = EnvStateKeys()

    class TargetStateKeys(typing.NamedTuple):
        orientation : str = 'Orientation'
        type : str = 'Type'
        img_path : str = 'Image Path'
        width : str = 'Target Width'
        height : str = 'Target Height'
        width_resolution : str = 'Width Resolution'
        height_resolution : str = 'Height Resolution'
    target_state_keys = TargetStateKeys()

    class TrainingResultsKeys(typing.NamedTuple):
        updated_at : str = 'Updated At'
        alignment_deviation : str = 'Alignment Deviation'
        predicted_normal : str = 'Predicted Normal'
        hausdorff_distance : str = 'Hausdorff Distance'
        is_training : str = 'Training'
        is_testing : str = 'Testing'
        is_evaluation : str = 'Evaluation'
    training_result_keys = TrainingResultsKeys()

    def __init__(self,
                    id : int,
                    heliostat : typing.Optional[str] = None,
                    ax1_steps : typing.Optional[torch.Tensor] = None,
                    ax2_steps : typing.Optional[torch.Tensor] = None,
                    created_at : typing.Optional[datetime.datetime] = None,

                    normal : typing.Optional[torch.Tensor] = None,
                    to_source : typing.Optional[torch.Tensor] = None,
                    source_pos : typing.Optional[torch.Tensor] = None,
                    
                    aimpoint : typing.Optional[torch.Tensor] = None,
                    target_state: typing.Optional[typing.Dict[str, any]] = None,

                    env_state : typing.Optional[typing.Dict[str, any]] = None,
                    training_results : typing.Optional[typing.Dict[str, any]] = None,
                    
                    desired_image : typing.Optional[torch.Tensor] = None,

                    dtype : torch.dtype = torch.get_default_dtype(),
                    device : torch.device = torch.device('cpu'),
                ):

        # if ((not torch.is_tensor(normal))
        #     and (not torch.is_tensor(to_source)) 
        #     and (not torch.is_tensor(source_pos))):
        #     raise Exception('Heliostat data either requires a normal vector or source vector and position')
        
        self._env_state_keys = HeliostatDataPoint.env_keys
        self._target_state_keys = HeliostatDataPoint.target_state_keys
        self._training_results_keys = HeliostatDataPoint.training_result_keys
        self._datetime_format = HeliostatDataPoint._datetime_format

        self._dtype = dtype
        self._device = device

        self.id = id
        
        self.heliostat = heliostat
        self.ax1_steps = ax1_steps
        self.ax2_steps = ax2_steps
        self.created_at = created_at

        # Optionals
        self.normal = normal
        self.aimpoint = aimpoint
        self.to_source = to_source
        self.source_pos = source_pos

        self.target_state = target_state if target_state else {}
        self._env_state = env_state if env_state else {}
        self.training_results = training_results if training_results else {}
        
        self.desired_image = desired_image
        
    def sourceAzim(self, heliostat_pos : typing.Optional[torch.Tensor] = None, to_deg : bool = False) -> torch.Tensor:
        to_source = None
        if (not torch.is_tensor(self.to_source)):
            raise Exception('Method not yet implemented!')
        else:
            to_source = self.to_source

        # if not torch.is_tensor(to_source):
        #     to_source = self.source_pos - self.heliostat_pos
        #     to_source = to_source / to_source.norm()

        angle = -torch.arctan2(to_source[COORDS.E], - to_source[COORDS.N]) + torch.pi
        if to_deg:
            angle = torch.rad2deg(angle)

        return angle

    def sourceElev(self, heliostat_pos : typing.Optional[torch.Tensor] = None, to_deg : bool = False) -> torch.Tensor:
        to_source = None
        if (not torch.is_tensor(self.to_source)):
            raise Exception('Method not yet implemented!')
        else:
            to_source = self.to_source

        # if not torch.is_tensor(to_source):
        #     to_source = self.source_pos - self.heliostat_pos
        #     to_source = to_source / to_source.norm()
        angle =  torch.arcsin(to_source[COORDS.U])
        if to_deg:
            angle = torch.rad2deg(angle)


        return angle

    def setElapsedStart(self, elapsed_start : datetime.datetime):
        self._env_state[self._env_state_keys.elapsed_start] = elapsed_start.strftime(self._datetime_format)

    def setElapsedEnd(self, elapsed_end : datetime.datetime):
        self._env_state[self._env_state_keys.elapsed_end] = elapsed_end.strftime(self._datetime_format)

    def elapsed_start(self, to_str: bool = False) -> typing.Optional[typing.Union[str, datetime.datetime]]:
        if self._env_state_keys.elapsed_start in self._env_state.keys():
            if to_str:
                elapsed_start = self._env_state[self._env_state_keys.elapsed_start]
            else:
                elapsed_start = datetime.datetime.strptime(self._env_state[self._env_state_keys.elapsed_start], self._datetime_format)
            return elapsed_start
        else:
            return None

    def elapsed_end(self, to_str: bool = False) -> typing.Optional[typing.Union[str, datetime.datetime]]:

        if self._env_state_keys.elapsed_end in self._env_state.keys():
            if to_str:
                elapsed_end = self._env_state[self._env_state_keys.elapsed_end]
            else:
                elapsed_end = datetime.datetime.strptime(self._env_state[self._env_state_keys.elapsed_end], self._datetime_format)
            return elapsed_end
        else:
            return None

    def env_state(self) -> typing.Dict[str, typing.Union[torch.Tensor, any]]:
        filled_env_state = {}

        # elapsed time
        filled_env_state[self._env_state_keys.month] = torch.tensor(self.created_at.month, dtype=self._dtype, device=self._device)

        if not self._env_state_keys.elapsed_days in self._env_state.keys():
            elapsed_start = self.elapsed_start()
            elapsed_end = self.elapsed_end()

            if elapsed_start:
                filled_env_state[self._env_state_keys.elapsed_start] = self.elapsed_start(to_str=True)
                filled_env_state[self._env_state_keys.elapsed_days] = torch.tensor((self.created_at - elapsed_start).days, dtype=self._dtype, device=self._device)

                if elapsed_end:
                    filled_env_state[self._env_state_keys.elapsed_end] = self.elapsed_end(to_str=True)
                    filled_env_state[self._env_state_keys.elapsed_days] = torch.tensor((self.created_at - elapsed_start).days / (elapsed_end - elapsed_start).days, dtype=self._dtype, device=self._device)
        else:
            filled_env_state[self._env_state_keys.elapsed_days] = torch.tensor(self._env_state[self._env_state_keys.elapsed_days], dtype=self._dtype, device=self._device)
        return filled_env_state

    def setHausdorffDistance(self, hausdorff_distance: torch.Tensor):
        self.training_results[self._training_results_keys.hausdorff_distance] = hausdorff_distance

    def hausdorff_distance(self) -> typing.Optional[torch.Tensor]:
        if not self._training_results_keys.hausdorff_distance in self.training_results:
            return None

        hd = self.training_results[self._training_results_keys.hausdorff_distance]
        if not torch.is_tensor(hd):
            hd = torch.tensor(hd, dtype=self._dtype, device=self._device)

        return hd

    def setAlignmentDeviation(self, alignment_deviation: torch.Tensor):
        self.training_results[self._training_results_keys.alignment_deviation] = alignment_deviation

    def alignment_deviation(self) -> typing.Optional[torch.Tensor]:
        if not self._training_results_keys.alignment_deviation in self.training_results:
            return None

        return self.training_results[self._training_results_keys.alignment_deviation]

    def distanceToDataset(self, 
                             data_points : typing.Dict[int, any],
                             selected_data_points : typing.List[typing.List[int]] = [],
                             dist_method : typing.Callable = HAUSDORFF.distanceByAngle,
                             update_distance : bool = False,
                             num_nearest_neighbors : int = 1,
                             return_extras : bool = False,
                             ) -> typing.Union[torch.Tensor, any]:
        hd, neighbors = HAUSDORFF.distanceToDataset(data_point=self, 
                                         data_points=data_points,
                                         selected_data_points=selected_data_points,
                                         dist_method=dist_method,
                                         num_nearest_neighbors=num_nearest_neighbors,
                                        )
        if update_distance:
            self.setHausdorffDistance(hausdorff_distance=hd)

        if return_extras:
            return hd, neighbors
        else:
            return hd
        
    def desired_image(self):
        return self.desired_image 
    
    def desired_concentrator_normal(self):
        return self.normal
    
    def sun_directions(self):
        return self.to_source
    