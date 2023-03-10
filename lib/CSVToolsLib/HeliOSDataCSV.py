import pandas as pd
import torch
import datetime
import matplotlib.pyplot as plt

# global definitions
E = 0
N = 1
U = 2
W = 0
H = 1
AXIS_1 = 0
AXIS_2 = 1

class HeliOSDataCSV:
    _sep = ';' # csv data seperator

    # source
    _source_vec_e = 'SunPosE'
    _source_vec_n = 'SunPosN'
    _source_vec_u = 'SunPosU'

    # target
    _target_center_e = 'TargetOffsetE'
    _target_center_n = 'TargetOffsetN'
    _target_center_u = 'TargetOffsetU'

    # heliostat
    _axis_steps_1 = 'Axis1MotorPosition'
    _axis_steps_2 = 'Axis2MotorPosition'
    _heliostat_id = 'HeliostatId'
    _heliostat_name = 'Name'

    # score
    _score = 'LastScore'

    # date and time
    _date_time = 'CreatedAt'
    
    _columns = [_heliostat_name,
                _source_vec_e,
                _source_vec_n,
                _source_vec_u,
                _target_center_e,
                _target_center_n,
                _target_center_u,
                _axis_steps_1,
                _axis_steps_2,
                _score,
                ]

    def __init__(self, dtype=torch.get_default_dtype(), device=torch.device('cpu')):
        self._device = device
        self._dtype = dtype
        self._data = None

    ##################
    #-    Reading   -#
    ##################
    def readCSV(self, csv_path, heliostat_name: str = None, date_time_range = None):
        self._data = pd.read_csv(csv_path, sep = self._sep)
        
        if not (heliostat_name is None):
            indices = []
            names = self.heliostat_names()
            dates = self.date()
            
            if not (dates is None or date_time_range is None):
                for index, (name, dt) in enumerate(zip(names, dates)):
                    if name == heliostat_name and self.validateDateInRange(dt=dt, date_time_range=date_time_range):
                        indices.append(index)
            else:
                for index, name in enumerate(names):
                    if name == heliostat_name:
                        indices.append(index)

            self._data = self._data.iloc[indices, :]

        print('- got data of shape: ' + str(self._data.shape))
        return self._data.shape[0]

    def validateDateInRange(self, dt, date_time_range):
        return dt >= date_time_range[0] and dt < date_time_range[1]

    # source
    def source_vec(self):
        return [torch.tensor([e, n, u], dtype=self._dtype, device=self._device) if not (e is None or n is None or u is None)
                else None
                for e, n, u 
                in zip(self._data[self._source_vec_e], self._data[self._source_vec_n], self._data[self._source_vec_u])]

    # target
    def target_center(self):
        return [torch.tensor([e, n, u], dtype=self._dtype, device=self._device) 
                for e, n, u 
                in zip(self._data[self._target_center_e], self._data[self._target_center_n], self._data[self._target_center_u])]

    # heliostat
    def axis_steps(self):
        return [torch.tensor([axis_1, axis_2], dtype=self._dtype, device=self._device) 
                for axis_1, axis_2 
                in zip(self._data[self._axis_steps_1], self._data[self._axis_steps_2])]

    def heliostat_names(self):
        return self._data[self._heliostat_name]

    # date and time
    def date(self):
        return [datetime.datetime.strptime(dt, "%Y-%m-%d %H:%M:%S")
                for dt in self._data[self._date_time]]
    
    ##################
    #-    Writing   -#
    ##################
    def addDataRow(self,
                   source_vec: torch.Tensor,
                   target_pos: torch.Tensor,
                   axis_steps: torch.Tensor,
                   heliostat_name: str,
                   ) -> None:
        data = [heliostat_name, 
                source_vec[0].detach().numpy(), source_vec[1].detach().numpy(), source_vec[2].detach().numpy(),
                target_pos[0].detach().numpy(), target_pos[1].detach().numpy(), target_pos[2].detach().numpy(),
                axis_steps[0].detach().numpy(), axis_steps[1].detach().numpy(),
                ]
        
        if self._data is None:
            self._data = pd.DataFrame(columns = self._columns)
            
        self._data.loc[len(self._data.index)] = data
        
    def writeCSV(self, csv_path):
        print('writing: ' + str(self._data.shape[0]) + 'columns')
        self._data.save(csv_path, sep=self._sep)

if __name__ == '__main__':
    csv = HeliOSDataCSV()
    csv_path = '/Users/Synhelion/Downloads/calibdata_max.csv'
    heliostat_name = 'AJ.23'
    start_date = datetime.datetime(year=2022, month=6, day=1).date()
    end_date = datetime.datetime(year=2022, month=7, day=1).date()
    csv.readCSV(csv_path=csv_path, heliostat_name=heliostat_name, date_time_range=[start_date, end_date])