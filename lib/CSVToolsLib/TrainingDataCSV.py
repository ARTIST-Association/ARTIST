import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import normalize

import datetime
import os
import yaml
from yaml.loader import SafeLoader
from contextlib import redirect_stdout
import sys

lib_dir = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(lib_dir)

lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir))
sys.path.append(lib_dir)

from HeliOSDataCSV import HeliOSDataCSV
from defaults import get_cfg_defaults, load_config_file

# global definitions
E = 0
N = 1
U = 2
W = 0
H = 1
AXIS_1 = 0
AXIS_2 = 1

class TrainingDataCSV:
    _sep = ';' # csv data seperator

    # source
    _source_pos_e = 'Source Pos E [m]'
    _source_pos_n = 'Source Pos N [m]'
    _source_pos_u = 'Source Pos U [m]'
    _source_vec_e = 'Source Vec E [m]'
    _source_vec_n = 'Source Vec N [m]'
    _source_vec_u = 'Source Vec U [m]'
    _source_diameter = 'Source Diameter [m]'

    # target
    _target_center_e = 'Target Center E [m]'
    _target_center_n = 'Target Center N [m]'
    _target_center_u = 'Target Center U [m]'
    _target_center_error_e = 'Target Center Error E [m]'
    _target_center_error_n = 'Target Center Error N [m]'
    _target_center_error_u = 'Target Center Error U [m]'
    _target_normal_e = 'Target Normal E [m]'
    _target_normal_n = 'Target Normal N [m]'
    _target_normal_u = 'Target Normal U [m]'
    _target_side_e = 'Target Side E [m]'
    _target_side_n = 'Target Side N [m]'
    _target_side_u = 'Target Side U [m]'
    _target_up_e = 'Target Up E [m]'
    _target_up_n = 'Target Up N [m]'
    _target_up_u = 'Target Up U [m]'
    _target_width = 'Target Width [m]'
    _target_height = 'Target Height [m]'
    _target_img_path = 'Target Image'

    # heliostat
    _axis_steps_1 = 'Axis 1 [Steps]'
    _axis_steps_2 = 'Axis 2 [Steps]'

    _columns = [
                _source_pos_e, _source_pos_n, _source_pos_u,
                _source_vec_e, _source_vec_n, _source_vec_u,
                _source_diameter,
                _target_center_e, _target_center_n, _target_center_u,
                _target_center_error_e, _target_center_error_n, _target_center_error_u,
                _target_normal_e, _target_normal_n, _target_normal_u,
                _target_side_e, _target_side_n, _target_side_u,
                _target_up_e, _target_up_n, _target_up_u,
                _target_width, _target_height,
                _target_img_path,
                _axis_steps_1, _axis_steps_2,
    ]

    def __init__(self, device=torch.device('cpu')):
        self._device = device
        self._data = None

    ##################
    #-    Reading   -#
    ##################
    def readCSV(self, csv_path, shuffle=False):
        self._data = pd.read_csv(csv_path, sep = self._sep)
        if shuffle:
            self._data = self._data.sample(frac=1)
        print('- got data of shape: ' + str(self._data.shape))

    # source
    def source_position(self):
        return [torch.tensor([e, n, u], device=self._device) if not (e is None or n is None or u is None)
                else None
                for e, n, u 
                in zip(self._data[self._source_pos_e], self._data[self._source_pos_n], self._data[self._source_pos_u])]

    def source_vec(self):
        return [torch.tensor([e, n, u], device=self._device) if not (e is None or n is None or u is None)
                else None
                for e, n, u 
                in zip(self._data[self._source_vec_e], self._data[self._source_vec_n], self._data[self._source_vec_u])]

    def source_diameter(self):
        return torch.tensor(self._data[self._solar_diameter], device=self._device)

    # target
    def target_center(self):
        return [torch.Tensor([e, n, u], device=self._device) 
                for e, n, u 
                in zip(self._data[self._target_center_e], self._data[self._target_center_n], self._data[self._target_center_u])]

    def target_center_error(self):
        return [torch.Tensor([e, n, u], device=self._device) 
                for e, n, u 
                in zip(self._data[self._target_center_error_e], self._data[self._target_center_error_n], self._data[self._target_center_error_u])]

    def target_normal(self):
        return [torch.Tensor([e, n, u], device=self._device) 
                for e, n, u 
                in zip(self._data[self._target_normal_e], self._data[self._target_normal_n], self._data[self._target_normal_u])]

    def target_width(self):
        return [w for w in self._data[self._target_width]]

    def target_height(self):
        return [h for h in self._data[self._target_height]]

    def target_image_paths(self):
        return self._data[self._target_img_path]

    # heliostat
    def axis_steps(self):
        return [torch.Tensor([axis_1, axis_2]) 
                for axis_1, axis_2 
                in zip(self._data[self._axis_steps_1], self._data[self._axis_steps_2])]

    ##################
    #-    Writing   -#
    ##################
    def writeCSVFromNumPoints(self, csv_path, num_points = -1):
        data_increment = int(self._data.shape[0] / num_points)
        data_increment = data_increment if data_increment > 0 else 1

        if data_increment > 1:

            selected_indices = []
            not_selected_indices = []

            for index in range(self._train_data.shape[0]):
                if index != 0 and (index % data_increment == 0):
                    selected_indices.append(index)
                else:
                    not_selected_indices.append(index)

            self.writeCSVFromIndices(csv_path=csv_path, indices=selected_indices)
            return not_selected_indices

        else:
            print('writing: ' + str(self._data.shape[0]) + 'columns')
            self._data.to_csv(csv_path, sep=self._sep)
            return None

    def writeCSVFromIndices(self, csv_path, indices):
        csv_export = self._data.iloc[indices, :]
        print('writing: ' + str(csv_export.shape[0]) + 'columns')
        csv_export.save(csv_path)

    def addDataRow(self, 
                    target_center : torch.Tensor,
                    target_width : float, 
                    target_height : float,
                    axis_steps_1 : torch.Tensor, 
                    axis_steps_2 : torch.Tensor,
                    target_center_error : torch.Tensor = torch.zeros(3),
                    target_image : str = None,
                    target_normal : torch.Tensor = torch.tensor([0, -1, 0]), 
                    target_side : torch.Tensor = torch.tensor([1, 0, 0]),
                    target_up : torch.Tensor = torch.tensor([0, 0, 1]),
                    source_pos : torch.Tensor = None, 
                    source_vec : torch.Tensor = None, 
                    source_diameter : float = 1,
                    ):
        assert (not (source_pos is None and source_vec is None)), 'either source position or vector must be given!'

        target_center = target_center + target_center_error

        if source_pos is None:
            data = [
                None, None, None,
                source_vec[E].detach().numpy(), source_vec[N].detach().numpy(), source_vec[U].detach().numpy(),
                source_diameter,
                target_center[E].detach().numpy(), target_center[N].detach().numpy(), target_center[U].detach().numpy(),
                target_center_error[E].detach().numpy(), target_center_error[N].detach().numpy(), target_center_error[U].detach().numpy(),
                target_normal[E].detach().numpy(), target_normal[N].detach().numpy(), target_normal[U].detach().numpy(),
                target_side[E].detach().numpy(), target_side[N].detach().numpy(), target_side[U].detach().numpy(),
                target_up[E].detach().numpy(), target_up[N].detach().numpy(), target_up[U].detach().numpy(),
                target_width, target_height,
                target_image,
                axis_steps_1.detach().numpy(), axis_steps_2.detach().numpy()
            ]
        else:
            data = [
                source_pos[E].detach().numpy(), source_pos[N].detach().numpy(), source_pos[U].detach().numpy(),
                None, None, None,
                source_diameter,
                target_center[E].detach().numpy(), target_center[N].detach().numpy(), target_center[U].detach().numpy(),
                target_center_error[E].detach().numpy(), target_center_error[N].detach().numpy(), target_center_error[U].detach().numpy(),
                target_normal[E].detach().numpy(), target_normal[N].detach().numpy(), target_normal[U].detach().numpy(),
                target_side[E].detach().numpy(), target_side[N].detach().numpy(), target_side[U].detach().numpy(),
                target_up[E].detach().numpy(), target_up[N].detach().numpy(), target_up[U].detach().numpy(),
                target_width, target_height,
                target_image,
                axis_steps_1.detach().numpy(), axis_steps_2.detach().numpy()
            ]
            

        if self._data is None:
            self._data = pd.DataFrame(columns = self._columns)
        
        self._data.loc[len(self._data.index)] = data

##################
#-     Main     -#
##################
def main():
    heliostat_names = ['AJ.23', 'AM.35', 'AM.42', 'AM.43']
    kinematic_types = ['Heliokon']
    actuator_1_params = [
                        [0.0775,0.3353,0.3381],
                        [0.0798,0.3353,0.3381],
                        [0.0713,0.3353,0.3381],
                        [0.0798,0.3353,0.3381],
    ]
    actuator_2_params = [
                        [0.0766,0.3408,0.3191],
                        [0.0782,0.3408,0.3191],
                        [0.0787,0.3408,0.3191],
                        [0.0757,0.3408,0.3191],
    ]
    positions = [
        [-57.2, 66.4, 88.729],
        [-4.4, 80.3, 88.735],
        [26.4, 80.3, 88.751],
        [30.8, 80.3, 88.753],
    ]
    start_dates_train = [datetime.datetime(year=2022, month=8, day=1).date(),
                         # datetime.datetime(year=2022, month=7, day=1).date(),
                         # datetime.datetime(year=2022, month=6, day=1).date(),
                         # datetime.datetime(year=2022, month=5, day=1).date(),
                         # datetime.datetime(year=2022, month=4, day=1).date(),
                         # datetime.datetime(year=2022, month=3, day=1).date(),
                         # datetime.datetime(year=2022, month=2, day=1).date(),
                         datetime.datetime(year=2022, month=1, day=1).date(),
                         #datetime.datetime(year=2022, month=9, day=1).date(),
                         #datetime.datetime(year=2022, month=1, day=1).date()
                         ]
    end_dates_train = [datetime.datetime(year=2022, month=9, day=1).date(),
                       # datetime.datetime(year=2022, month=9, day=1).date(),
                       # datetime.datetime(year=2022, month=9, day=1).date(),
                       # datetime.datetime(year=2022, month=9, day=1).date(),
                       # datetime.datetime(year=2022, month=9, day=1).date(),
                       # datetime.datetime(year=2022, month=9, day=1).date(),
                       # datetime.datetime(year=2022, month=9, day=1).date(),
                       datetime.datetime(year=2022, month=9, day=1).date(),
                       #datetime.datetime(year=2022, month=11, day=1).date(),
                       #datetime.datetime(year=2022, month=11, day=1).date()
                       ]
    eval_dates_start = [datetime.datetime(year=2022, month=9, day=1).date(), 
                        # datetime.datetime(year=2022, month=9, day=1).date(),
                        # datetime.datetime(year=2022, month=10, day=1).date()
                        ]
    eval_dates_end = [datetime.datetime(year=2022, month=11, day=1).date(),
                      # datetime.datetime(year=2022, month=10, day=1).date(),
                      # datetime.datetime(year=2022, month=11, day=1).date()
                      ]
    #meas_errors = [0, 0.5, 1]
    meas_errors = [0.5, 1]
    seeds = [1,42,130,1133,8900,4012,2454,7898,6654,5365]

    default_config_path = '/home/user/Desktop/moritz/V11 - New Architecture/TestingConfigs/_config_moritz_am42.yaml'
    data_path = '/home/user/Desktop/moritz/V11 - New Architecture/ExperimentData/calibdata_max.csv'
    
    output_dir = '/home/user/Desktop/moritz/V11 - New Architecture/ExperimentData'
    for hs_name, ac1_params, ac2_params, pos in zip(heliostat_names, actuator_1_params, actuator_2_params, positions):
        for kin_type in kinematic_types:
            for me in meas_errors:
                for sdt, edt in zip(start_dates_train, end_dates_train):
                    for sde, ede in zip(eval_dates_start, eval_dates_end):
                        for s in seeds:

                            # paths
                            exp_name = hs_name + '_' + kin_type + '_train_' + str(sdt.month) + 'to' + str(edt.month) + '_eval_' + str(sde.month) + 'to' + str(ede.month) + '_' + str(me) + 'cm_' + str(s)
                            train_path = os.sep.join([output_dir, exp_name + '_train' + '.csv'])
                            eval_path = os.sep.join([output_dir, exp_name + '_eval' + '.csv'])
                            config_path = os.sep.join([output_dir, exp_name + '.yaml'])

                            # config
                            cfg = get_cfg_defaults()
                            opts = [
                                "ALIGNMENT.KINEMATIC.TYPE", kin_type,
                                "ALIGNMENT.KINEMATIC.HELIOKON.USE_PARAMETERS", True,
                                "ALIGNMENT.KINEMATIC.HELIOKON.ACTUATOR_1.PARAM_B", ac1_params[0],
                                "ALIGNMENT.KINEMATIC.HELIOKON.ACTUATOR_1.PARAM_C", ac1_params[1],
                                "ALIGNMENT.KINEMATIC.HELIOKON.ACTUATOR_1.PARAM_D", ac1_params[2],
                                "ALIGNMENT.KINEMATIC.HELIOKON.ACTUATOR_2.PARAM_B", ac2_params[0],
                                "ALIGNMENT.KINEMATIC.HELIOKON.ACTUATOR_2.PARAM_C", ac2_params[1],
                                "ALIGNMENT.KINEMATIC.HELIOKON.ACTUATOR_2.PARAM_D", ac2_params[2],
                                "EXPERIMENT_NAME", exp_name,
                                "H.DEFLECT_DATA.POSITION_ON_FIELD", pos,
                                "TRAIN.EPOCHS", 400,
                                "TRAIN.DATA_LIST.TOGGLE", True,
                                "TRAIN.DATA_LIST.PATH", train_path,
                                "TRAIN.DATA_LIST.EVAL_PATH", eval_path,
                                "TRAIN.EARLY_STOPPING.PATIENCE", 6,
                                "USE_GPU", False,
                                "USE_NURBS", False
                            ]
                            if not (default_config_path is None):
                                cfg.merge_from_file(default_config_path)
                            cfg.merge_from_list(opts)

                            # train data reader
                            helios_csv_train = HeliOSDataCSV()
                            num_points_train = helios_csv_train.readCSV(csv_path=data_path,
                                            heliostat_name=hs_name,
                                            date_time_range=[sdt, edt]
                                            )
                            train_csv = TrainingDataCSV()

                            torch.manual_seed(s)
                            offsets = (torch.rand((num_points_train,3)) - 0.5) * 2
                            offsets = normalize(offsets, dim=1)
                            # print(offsets)
                            offsets = offsets * me

                            for (source_vec, target_center, axis_steps, target_error) in zip(helios_csv_train.source_vec(), helios_csv_train.target_center(), helios_csv_train.axis_steps(), offsets):
        
                                train_csv.addDataRow(
                                                    target_center=target_center,
                                                    target_width = 10, target_height = 10,
                                                    axis_steps_1 = axis_steps[AXIS_1],
                                                    axis_steps_2 = axis_steps[AXIS_2],
                                                    target_image='Transform.png',
                                                    target_center_error=target_error,
                                                    source_vec=source_vec
                                                    )

                            train_csv.writeCSVFromNumPoints(csv_path=train_path)

                            # eval data reader
                            helios_csv_eval = HeliOSDataCSV()
                            num_points_eval = helios_csv_eval.readCSV(csv_path=data_path,
                                            heliostat_name=hs_name,
                                            date_time_range=[sde, ede]
                                            )
                            eval_csv = TrainingDataCSV()

                            for (source_vec, target_center, axis_steps) in zip(helios_csv_eval.source_vec(), helios_csv_eval.target_center(), helios_csv_eval.axis_steps()):
        
                                eval_csv.addDataRow(
                                                    target_center=target_center,
                                                    target_width = 10, target_height = 10,
                                                    axis_steps_1 = axis_steps[AXIS_1],
                                                    axis_steps_2 = axis_steps[AXIS_2],
                                                    target_image='Transform.png',
                                                    source_vec=source_vec
                                                    )

                            eval_csv.writeCSVFromNumPoints(csv_path=eval_path)


if __name__ == '__main__':
    main()