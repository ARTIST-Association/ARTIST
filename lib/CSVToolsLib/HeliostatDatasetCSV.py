import pandas as pd
import torch
import datetime
import os
import sys
import typing
import json

lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(lib_dir)
import CoordinateSystemsLib.ExtendedCoordinates as COORDS
from HeliostatTrainingLib.HeliostatDatapoint import HeliostatDataPoint

class HeliostatDatasetCSV:
    _sep = ';' # csv data seperator
    _created_at_format = "%Y-%m-%d %H:%M:%S"

    class ColumnNames(typing.NamedTuple):
        # ids
        data_point_id = 'DataPoind ID' # int (-> maybe better str?)
        heliostat_name = 'Heliostat Name' # str

        # heliostat
        axis_steps_1 = 'ActuatorPosition_1' # int
        axis_steps_2 = 'ActuatorPosition_2' # int

        # normal
        normal_e = 'Normal_E' # float
        normal_n = 'Normal_N' # float
        normal_u = 'Normal_U'  # float

        # source
        to_source_e = 'ToSource_E' # float
        to_source_n = 'ToSource_N' # float
        to_source_u = 'ToSource_U' # float

        source_pos_e = 'SourcePos_E' # float
        source_pos_n = 'SourcePos_N' # float
        source_pos_u = 'SourcePos_U' # float

        # target
        aimpoint_e = 'Aimpoint_E' # float
        aimpoint_n = 'Aimpoint_N' # float
        aimpoint_u = 'Aimpoint_U' # float

        target_state = 'Target State' # Dict: Orientation, Type (e.g. HelioPoint, CameraTarget, UltraCalib), Img Path, Dimensions

        # environment
        created_at = 'Created At' # str: YYYY-MM-DD hh:mm:ss
        env_state = 'Env. State' # Dict: time_elapsed_start, time_elapsed_end, etc.

        # training results
        training_results = 'Training Results' # Dict: data type (train, test, eval), hausdorff distance, predicted normal, alignment deviation, training system

    _column_names = ColumnNames()
    
    _columns = [_column_names.data_point_id,
                _column_names.heliostat_name,
                _column_names.axis_steps_1,
                _column_names.axis_steps_2,
                _column_names.normal_e,
                _column_names.normal_n,
                _column_names.normal_u,
                _column_names.to_source_e,
                _column_names.to_source_n,
                _column_names.to_source_u,
                _column_names.source_pos_e,
                _column_names.source_pos_n,
                _column_names.source_pos_u,
                _column_names.aimpoint_e,
                _column_names.aimpoint_n,
                _column_names.aimpoint_u,
                _column_names.target_state,
                _column_names.created_at,
                _column_names.env_state,
                _column_names.training_results,
                ]

    def __init__(self, dtype=torch.get_default_dtype(), device=torch.device('cpu')):
        self._device = device
        self._dtype = dtype
        self._data = None
        self._column_names = HeliostatDatasetCSV._column_names

    ##################
    #-    Reading   -#
    ##################
    def readCSV(self, csv_path):
        if os.path.isfile(csv_path):
            self._data = pd.read_csv(csv_path, sep = self._sep)
        else:
            self._data = pd.DataFrame(columns = self._columns)
            print('- got data of shape: ' + str(self._data.shape))
        return self._data

    def _data_point_ids(self) -> typing.List[int]:
        return self._data[self._column_names.data_point_id]

    def _heliostat_names(self) -> typing.List[str]:
        return self._data[self._column_names.heliostat_name]

    def _actuator_positions_1(self) -> typing.List[torch.Tensor]:
        return [torch.tensor(ap, dtype=self._dtype, device=self._device) 
                for ap in self._data[self._column_names.axis_steps_1]]

    def _actuator_positions_2(self) -> typing.List[torch.Tensor]:
        return [torch.tensor(ap, dtype=self._dtype, device=self._device) 
                for ap in self._data[self._column_names.axis_steps_2]]

    def _normals(self) -> typing.List[typing.Optional[torch.Tensor]]:
        if self._column_names.normal_e in self._data.columns:
            return [torch.tensor([e,n,u], dtype=self._dtype, device=self._device)
                    for e,n,u in zip(self._data[self._column_names.normal_e], 
                                    self._data[self._column_names.normal_n], 
                                    self._data[self._column_names.normal_u])]
        else:
            return [None for i in range(self._data.shape[0])] 

    def _to_source_vecs(self) -> typing.List[typing.Optional[torch.Tensor]]:
        if self._column_names.to_source_e in self._data.columns:
            return [torch.tensor([e,n,u], dtype=self._dtype, device=self._device)
                    for e,n,u in zip(self._data[self._column_names.to_source_e], 
                                    self._data[self._column_names.to_source_n], 
                                    self._data[self._column_names.to_source_u])]
        else:
            return [None for i in range(self._data.shape[0])] 

    def _source_positions(self) -> typing.List[typing.Optional[torch.Tensor]]:
        if self._column_names.source_pos_e in self._data.columns:
            return [torch.tensor([e,n,u], dtype=self._dtype, device=self._device)
                    for e,n,u in zip(self._data[self._column_names.source_pos_e], 
                                    self._data[self._column_names.source_pos_n], 
                                    self._data[self._column_names.source_pos_u])]
        else:
            return [None for i in range(self._data.shape[0])] 

    def _aimpoints(self) -> typing.List[typing.Optional[torch.Tensor]]:
        if self._column_names.aimpoint_e in self._data.columns:
            return [torch.tensor([e,n,u], dtype=self._dtype, device=self._device)
                    for e,n,u in zip(self._data[self._column_names.aimpoint_e], 
                                    self._data[self._column_names.aimpoint_n], 
                                    self._data[self._column_names.aimpoint_u])]
        else:
            return [None for i in range(self._data.shape[0])] 

    def _target_states(self) -> typing.List[typing.Optional[typing.Dict[str, any]]]:
        if self._column_names.target_state in self._data.columns:
            return [json.loads(json_str) if json_str else None for json_str in self._data[self._column_names.target_state]]
        else:
            return [None for i in range(self._data.shape[0])] 

    def _created_at_dates(self) -> typing.List[datetime.datetime]:
        return [datetime.datetime.strptime(dt, self._created_at_format)
                for dt in self._data[self._column_names.created_at]]

    def _env_states(self) -> typing.List[typing.Optional[typing.Dict[str, any]]]:
        if self._column_names.env_state in self._data.columns:
            return [json.loads(json_str) if json_str else None for json_str in self._data[self._column_names.env_state]]
        else:
            return [None for i in range(self._data.shape[0])] 

    def _training_results(self) -> typing.List[typing.Optional[typing.Dict[str, any]]]:
        if self._column_names.training_results in self._data.columns:
            return [json.loads(json_str) if json_str else None for json_str in self._data[self._column_names.training_results]]
        else:
            return [None for i in range(self._data.shape[0])] 

    def dataPoints(self) -> typing.Dict[int, HeliostatDataPoint]:
        data_points = {}
        for (dp_id, hs_name, ap_1, ap_2, normal, to_source, source_pos,
             aimpoint, target_state, created_at, env_state, training_results) \
             in zip(self._data_point_ids(), self._heliostat_names(), self._actuator_positions_1(), self._actuator_positions_2(),
                    self._normals(), self._to_source_vecs(), self._source_positions(), self._aimpoints(),
                    self._target_states(), self._created_at_dates(), self._env_states(), self._training_results()):

            normal = normal if (torch.is_tensor(normal) and not torch.any(torch.isnan(normal))) else None
            to_source = to_source if (torch.is_tensor(to_source) and not torch.any(torch.isnan(to_source))) else None
            source_pos = source_pos if (torch.is_tensor(source_pos) and not torch.any(torch.isnan(source_pos))) else None

            dp = HeliostatDataPoint(id = dp_id,
                                     heliostat = hs_name,
                                     ax1_steps = ap_1,
                                     ax2_steps = ap_2,
                                     normal = normal,
                                     to_source = to_source,
                                     source_pos = source_pos,
                                     aimpoint = aimpoint,
                                     target_state = target_state,
                                     created_at = created_at,
                                     env_state = env_state,
                                     training_results = training_results,
                                     dtype = self._dtype,
                                     device = self._device,
                                    )
            data_points[dp_id] = dp

        return data_points

    def addDataRow(self, data_point: HeliostatDataPoint):
        normal_e = data_point.normal[COORDS.E].detach().numpy() if torch.is_tensor(data_point.normal) else None
        normal_n = data_point.normal[COORDS.N].detach().numpy() if torch.is_tensor(data_point.normal) else None
        normal_u = data_point.normal[COORDS.U].detach().numpy() if torch.is_tensor(data_point.normal) else None
        to_source_e = data_point.to_source[COORDS.E].detach().numpy() if torch.is_tensor(data_point.to_source) else None
        to_source_n = data_point.to_source[COORDS.N].detach().numpy() if torch.is_tensor(data_point.to_source) else None
        to_source_u = data_point.to_source[COORDS.U].detach().numpy() if torch.is_tensor(data_point.to_source) else None
        source_pos_e = data_point.source_pos[COORDS.E].detach().numpy() if torch.is_tensor(data_point.source_pos) else None
        source_pos_n = data_point.source_pos[COORDS.N].detach().numpy() if torch.is_tensor(data_point.source_pos) else None
        source_pos_u = data_point.source_pos[COORDS.U].detach().numpy() if torch.is_tensor(data_point.source_pos) else None
        aimpoint_e = data_point.aimpoint[COORDS.E].detach().numpy() if torch.is_tensor(data_point.aimpoint) else None
        aimpoint_n = data_point.aimpoint[COORDS.N].detach().numpy() if torch.is_tensor(data_point.aimpoint) else None
        aimpoint_u = data_point.aimpoint[COORDS.U].detach().numpy() if torch.is_tensor(data_point.aimpoint) else None

        target_state = json.dumps(data_point.target_state)
        created_at = data_point.created_at.strftime(self._created_at_format)
        env_state = data_point.env_state().copy()
        for key in env_state:
            if isinstance(env_state[key], torch.Tensor):
                env_state[key] = env_state[key].item()
        env_state = json.dumps(env_state)

        training_results = data_point.training_results.copy()
        for key in training_results:
            if isinstance(training_results[key], torch.Tensor):
                training_results[key] = training_results[key].tolist()
        training_results = json.dumps(training_results)

        data = [data_point.id,
                data_point.heliostat,
                data_point.ax1_steps.detach().numpy(),
                data_point.ax2_steps.detach().numpy(),
                normal_e, normal_n, normal_u,
                to_source_e, to_source_n, to_source_u,
                source_pos_e, source_pos_n, source_pos_u,
                aimpoint_e, aimpoint_n, aimpoint_u,
                target_state, created_at, env_state, training_results
                ]

        if self._data is None:
            self._data = pd.DataFrame(columns = self._columns)
            
        self._data.loc[len(self._data.index)] = data

    def writeCSV(self, csv_path):
        if isinstance(self._data, pd.DataFrame):
            print('writing: ' + str(self._data.shape[0]) + 'columns')
            self._data.to_csv(csv_path, sep=self._sep)

class HeliOSDatasetCSV(HeliostatDatasetCSV):

    class ColumnNames(typing.NamedTuple):
        # ids
        data_point_id = 'id' # int (-> maybe better str?)
        heliostat_name = 'Name' # str

        # heliostat
        axis_steps_1 = 'Axis1MotorPosition' # int
        axis_steps_2 = 'Axis2MotorPosition' # int

        # normal
        normal_e = 'Normal_E' # float
        normal_n = 'Normal_N' # float
        normal_u = 'Normal_U'  # float

        # source
        to_source_e = 'SunPosE' # float
        to_source_n = 'SunPosN' # float
        to_source_u = 'SunPosU' # float

        source_pos_e = 'SourcePos_E' # float
        source_pos_n = 'SourcePos_N' # float
        source_pos_u = 'SourcePos_U' # float

        # target
        aimpoint_e = 'TargetOffsetE' # float
        aimpoint_n = 'TargetOffsetN' # float
        aimpoint_u = 'TargetOffsetU' # float

        target_state = 'Target State' # Dict: Orientation, Type (e.g. HelioPoint, CameraTarget, UltraCalib), Img Path, Dimensions

        # environment
        created_at = 'CreatedAt' # str: YYYY-MM-DD hh:mm:ss
        env_state = 'Env. State' # Dict: time_elapsed_start, time_elapsed_end, etc.

        # training results
        training_results = 'Training Results' # Dict: data type (train, test, eval), hausdorff distance, predicted normal, alignment deviation, training system

    _column_names = ColumnNames()
    
    _columns = [_column_names.data_point_id,
                _column_names.heliostat_name,
                _column_names.axis_steps_1,
                _column_names.axis_steps_2,
                _column_names.normal_e,
                _column_names.normal_n,
                _column_names.normal_u,
                _column_names.to_source_e,
                _column_names.to_source_n,
                _column_names.to_source_u,
                _column_names.source_pos_e,
                _column_names.source_pos_n,
                _column_names.source_pos_u,
                _column_names.aimpoint_e,
                _column_names.aimpoint_n,
                _column_names.aimpoint_u,
                _column_names.target_state,
                _column_names.created_at,
                _column_names.env_state,
                _column_names.training_results,
                ]

    def __init__(self, dtype=torch.get_default_dtype(), device=torch.device('cpu')):
        self._device = device
        self._dtype = dtype
        self._data = None
        self._column_names = HeliOSDatasetCSV._column_names