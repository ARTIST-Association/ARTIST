# system dependencies
import torch
import typing
import datetime
import sys
import os
import json
import copy

# local dependencies
module_dir = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(module_dir)
from HeliostatDatapoint import HeliostatDataPoint

lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(lib_dir)
import CoordinateSystemsLib.ExtendedCoordinates as COORDS
from CSVToolsLib.HeliostatDatasetCSV import HeliostatDatasetCSV

class HeliostatDataset:

    default_config_name = 'dataset_config.json'
    default_data_name = 'dataset_data.csv'

    class ConfigKeys(typing.NamedTuple):
        all : str = 'all'
        training : str = 'training'
        testing : str = 'testing'
        evaluation : str = 'evaluation'

        heliostats_list : str = 'heliostats'
        heliostat_pos : str = 'heliostat_pos'
        data_point_list : str = 'data points'
        date_range : str = 'date_range'
        source_azim_range : str = 'source_azim_range'
        source_elev_range : str = 'source_elev_range'
        actuator_1_range : str = 'actuator_1_range'
        actuator_2_range : str = 'actuator_2_range'
    config_keys = ConfigKeys()

    def __init__(self,
                 data_points : typing.Union[str, typing.Dict[int, HeliostatDataPoint]],
                 csv_input_reader_type : typing.Callable = HeliostatDatasetCSV,
                 dataset_config : typing.Dict[str, any] = {},
                 dtype : torch.dtype = torch.get_default_dtype(),
                 device : torch.device = torch.device('cpu'),
                ):

        self._dtype = dtype
        self._device = device

        self._config_keys = HeliostatDataset.config_keys

        self._dataset_config = dataset_config
        self._active_data_points : typing.List[int] = []
        self._training_data_points : typing.List[int] = []
        self._testing_data_points : typing.List[int] = []
        self._evaluation_data_points : typing.List[int] = []

        if isinstance(data_points, str):
            csv_input_reader = csv_input_reader_type(dtype=self._dtype, device = self._device)
            csv_input_reader.readCSV(csv_path=data_points)
            self._data_points = csv_input_reader.dataPoints()
        else:
            self._data_points = data_points.copy()

        self.setDataPoints(dataset_config=dataset_config)

    def setDataPoints(self, 
                      dataset_config : typing.Dict[str, any] = {},
                      num_nearest_neighbors : int = 1,
                      ):
        self._active_data_points : typing.List[int] = None
        self._training_data_points : typing.List[int] = None
        self._testing_data_points : typing.List[int] = None
        self._evaluation_data_points : typing.List[int] = None

        if dataset_config:
            self._active_data_points = self._reduceDatasetByConfig(dataset_keys=list(self._data_points.keys()), 
                                                                   dataset_config=dataset_config[self._config_keys.all])
            self._evaluation_data_points = self._reduceDatasetByConfig(dataset_keys=self._active_data_points, 
                                                                   dataset_config=dataset_config[self._config_keys.evaluation])
            self._testing_data_points = self._reduceDatasetByConfig(dataset_keys=self._active_data_points, 
                                                                   dataset_config=dataset_config[self._config_keys.testing])
            self._training_data_points = self._reduceDatasetByConfig(dataset_keys=self._active_data_points, 
                                                                   dataset_config=dataset_config[self._config_keys.training])

        if not self._active_data_points:
            self._active_data_points = list(self._data_points.keys())
        if not self._training_data_points:
            self._training_data_points = self._active_data_points

        self._updateHausdorffDistances(num_nearest_neighbors=num_nearest_neighbors)
        self._updateConfig()
        self._updateElapsed()

    def _reduceDatasetByConfig(self, 
                                dataset_keys : typing.List[int], 
                                dataset_config : typing.Dict[str, any],
                                ) -> typing.List[int]:
        
        accepted_keys = []
        # accept pre determined keys
        if self._config_keys.data_point_list in dataset_config:
            for key in dataset_config[self._config_keys.data_point_list]:
                if key in dataset_keys:
                    accepted_keys.append(key)
        
        else:
            # build restricted dataset
            for key in dataset_keys:
                heliostat = self._data_points[key].heliostat

                # datapoint can't be training, testing and/or evaluation at once
                if ((self._training_data_points and key in self._training_data_points)
                    or (self._testing_data_points and key in self._testing_data_points)
                    or (self._evaluation_data_points and key in self._evaluation_data_points)
                    ):
                    continue

                # check heliostat
                if (self._config_keys.heliostats_list in dataset_config
                    and not heliostat in dataset_config[self._config_keys.heliostats_list]):
                    continue 

                # check date range
                created_at = self._data_points[key].created_at
                if (self._config_keys.date_range in dataset_config
                    and dataset_config[self._config_keys.date_range][0]
                    and created_at <= datetime.datetime.strptime(dataset_config[self._config_keys.date_range][0], HeliostatDataPoint._datetime_format)
                    ):
                    dt = datetime.datetime.strptime(dataset_config[self._config_keys.date_range][0], HeliostatDataPoint._datetime_format)
                    continue

                if (self._config_keys.date_range in dataset_config
                    and dataset_config[self._config_keys.date_range][1]
                    and self._data_points[key].created_at >= datetime.datetime.strptime(dataset_config[self._config_keys.date_range][1], HeliostatDataPoint._datetime_format)
                    ):
                    continue

                # check source angles
                azim = torch.rad2deg(self._data_points[key].sourceAzim()).detach().numpy()
                if (self._config_keys.source_azim_range in dataset_config
                    and dataset_config[self._config_keys.source_azim_range][0]
                    and azim <= dataset_config[self._config_keys.source_azim_range][0]
                    ):
                    continue

                if (self._config_keys.source_azim_range in dataset_config
                    and dataset_config[self._config_keys.source_azim_range][1]
                    and azim >= dataset_config[self._config_keys.source_azim_range][1]
                    ):
                    continue

                elev = torch.rad2deg(self._data_points[key].sourceElev()).detach().numpy()
                if (self._config_keys.source_elev_range in dataset_config
                    and dataset_config[self._config_keys.source_elev_range][0]
                    and elev <= dataset_config[self._config_keys.source_elev_range][0]
                    ):
                    continue

                if (self._config_keys.source_elev_range in dataset_config
                    and dataset_config[self._config_keys.source_elev_range][1]
                    and elev >= dataset_config[self._config_keys.source_elev_range][1]
                    ):
                    continue

                # check actuators
                if (self._config_keys.actuator_1_range in dataset_config
                    and dataset_config[self._config_keys.actuator_1_range][0]
                    and self._data_points[key].ax1_steps.detach().numpy() <= dataset_config[self._config_keys.actuator_1_range][0]
                    ):
                    continue

                if (self._config_keys.actuator_1_range in dataset_config
                    and dataset_config[self._config_keys.actuator_1_range][1]
                    and self._data_points[key].ax1_steps.detach().numpy() >= dataset_config[self._config_keys.actuator_1_range][1]
                    ):
                    continue

                if (self._config_keys.actuator_2_range in dataset_config
                    and dataset_config[self._config_keys.actuator_2_range][0]
                    and self._data_points[key].ax2_steps.detach().numpy() <= dataset_config[self._config_keys.actuator_2_range][0]
                    ):
                    continue

                if (self._config_keys.actuator_2_range in dataset_config
                    and dataset_config[self._config_keys.actuator_2_range][1]
                    and self._data_points[key].ax2_steps.detach().numpy() >= dataset_config[self._config_keys.actuator_2_range][1]
                    ):
                    continue

                accepted_keys.append(key)

        return accepted_keys

    def _updateHausdorffDistances(self,
                                  num_nearest_neighbors: int = 1,
                                  ):
        # for key in self._training_data_points:
        #     self._data_points[key].distanceToDataset(data_points=self._data_points, 
        #                                              selected_data_points=[self._training_data_points],
        #                                              update_distance = True,
        #                                              num_nearest_neighbors=num_nearest_neighbors,
        #                                             )
        for key in self._testing_data_points:
            self._data_points[key].distanceToDataset(data_points=self._data_points, 
                                                     selected_data_points=[self._training_data_points],
                                                     update_distance = True,
                                                     num_nearest_neighbors=num_nearest_neighbors,
                                                    )
        for key in self._evaluation_data_points:
            self._data_points[key].distanceToDataset(data_points=self._data_points, 
                                                     selected_data_points=[self._training_data_points],
                                                     update_distance = True,
                                                     num_nearest_neighbors=num_nearest_neighbors,
                                                    )

    def _updateElapsed(self,
                        end_date : datetime.datetime = datetime.datetime(year=2024, month=1, day=1),
                        start_date : typing.Optional[datetime.datetime] = None,
                        ) :
        if not start_date:
            for dp in self._data_points.values():
                if not start_date or dp.created_at < start_date:
                    start_date = dp.created_at
        
        for key in self._data_points.keys():
            self._data_points[key].setElapsedStart(start_date)
            self._data_points[key].setElapsedEnd(end_date)

    def _updateConfig(self):
        self._dataset_config = {}
        
        config_dict = {
            self._config_keys.heliostats_list: [],
            self._config_keys.data_point_list: [],
            self._config_keys.date_range: [None, None],
            self._config_keys.source_azim_range: [None, None],
            self._config_keys.source_elev_range: [None, None],
            self._config_keys.actuator_1_range: [None, None],
            self._config_keys.actuator_2_range: [None, None],
        }
        self._dataset_config[self._config_keys.all] = copy.deepcopy(config_dict)
        self._dataset_config[self._config_keys.training] = copy.deepcopy(config_dict)
        self._dataset_config[self._config_keys.testing] = copy.deepcopy(config_dict)
        self._dataset_config[self._config_keys.evaluation] = copy.deepcopy(config_dict)

        self._dataset_config[self._config_keys.all][self._config_keys.data_point_list] = list(self._active_data_points)
        self._dataset_config[self._config_keys.training][self._config_keys.data_point_list] = list(self._training_data_points)
        self._dataset_config[self._config_keys.testing][self._config_keys.data_point_list] = list(self._testing_data_points)
        self._dataset_config[self._config_keys.evaluation][self._config_keys.data_point_list] = list(self._evaluation_data_points)

        dtc_list = [self._config_keys.all]
        dtc_list.append(self._config_keys.training)
        dtc_list.append(self._config_keys.testing)
        dtc_list.append(self._config_keys.evaluation)

        for key in self._active_data_points:

            for dtc in dtc_list:
                # datapoint id
                if not key in self._dataset_config[dtc][self._config_keys.data_point_list]:
                    continue

                # heliostat
                heliostat = self._data_points[key].heliostat
                if not heliostat in self._dataset_config[dtc][self._config_keys.heliostats_list]:
                    self._dataset_config[dtc][self._config_keys.heliostats_list].append(heliostat)

                # date range
                created_at = self._data_points[key].created_at
                if (not self._dataset_config[dtc][self._config_keys.date_range][0]
                    or created_at < datetime.datetime.strptime(self._dataset_config[dtc][self._config_keys.date_range][0], HeliostatDataPoint._datetime_format)
                    ):
                    self._dataset_config[dtc][self._config_keys.date_range][0] = created_at.strftime(HeliostatDataPoint._datetime_format)

                if (not self._dataset_config[dtc][self._config_keys.date_range][1]
                    or created_at > datetime.datetime.strptime(self._dataset_config[dtc][self._config_keys.date_range][1], HeliostatDataPoint._datetime_format)
                    ):
                    self._dataset_config[dtc][self._config_keys.date_range][1] = created_at.strftime(HeliostatDataPoint._datetime_format)
            
                # source angles
                azim = torch.rad2deg(self._data_points[key].sourceAzim()).item()
                if (not self._dataset_config[dtc][self._config_keys.source_azim_range][0]
                    or azim < self._dataset_config[dtc][self._config_keys.source_azim_range][0]
                    ):
                    self._dataset_config[dtc][self._config_keys.source_azim_range][0] = azim

                if (not self._dataset_config[dtc][self._config_keys.source_azim_range][1]
                    or azim > self._dataset_config[dtc][self._config_keys.source_azim_range][1]
                    ):
                    self._dataset_config[dtc][self._config_keys.source_azim_range][1] = azim

                elev = torch.rad2deg(self._data_points[key].sourceElev()).item()
                if (not self._dataset_config[dtc][self._config_keys.source_elev_range][0]
                    or elev < self._dataset_config[dtc][self._config_keys.source_elev_range][0]
                    ):
                    self._dataset_config[dtc][self._config_keys.source_elev_range][0] = elev

                if (not self._dataset_config[dtc][self._config_keys.source_elev_range][1]
                    or elev > self._dataset_config[dtc][self._config_keys.source_elev_range][1]
                    ):
                    self._dataset_config[dtc][self._config_keys.source_elev_range][1] = elev

                actuator_1 = self._data_points[key].ax1_steps.item()
                if (not self._dataset_config[dtc][self._config_keys.actuator_1_range][0]
                    or actuator_1 < self._dataset_config[dtc][self._config_keys.actuator_1_range][0]
                    ):
                    self._dataset_config[dtc][self._config_keys.actuator_1_range][0] = actuator_1

                if (not self._dataset_config[dtc][self._config_keys.actuator_1_range][1]
                    or actuator_1 > self._dataset_config[dtc][self._config_keys.actuator_1_range][1]
                    ):
                    self._dataset_config[dtc][self._config_keys.actuator_1_range][1] = actuator_1

                actuator_2 = self._data_points[key].ax2_steps.item()
                if (not self._dataset_config[dtc][self._config_keys.actuator_2_range][0]
                    or actuator_2 < self._dataset_config[dtc][self._config_keys.actuator_2_range][0]
                    ):
                    self._dataset_config[dtc][self._config_keys.actuator_2_range][0] = actuator_2

                if (not self._dataset_config[dtc][self._config_keys.actuator_2_range][1]
                    or actuator_2 > self._dataset_config[dtc][self._config_keys.actuator_2_range][1]
                    ):
                    self._dataset_config[dtc][self._config_keys.actuator_2_range][1] = actuator_2

    def trainingDataset(self) -> typing.Dict[int, HeliostatDataPoint]:
        dataset = {}
        for key in self._training_data_points:
            dataset[key] = self._data_points[key]
        return dataset

    def testingDataset(self) -> typing.Dict[int, HeliostatDataPoint]:
        dataset = {}
        for key in self._testing_data_points:
            dataset[key] = self._data_points[key]
        return dataset

    def evaluationDataset(self) -> typing.Dict[int, HeliostatDataPoint]:
        dataset = {}
        for key in self._evaluation_data_points:
            dataset[key] = self._data_points[key]
        return dataset

    def activeDataset(self) -> typing.Dict[int, HeliostatDataPoint]:
        dataset = self.trainingDataset() | self.testingDataset()
        dataset = dataset | self.evaluationDataset()
        return dataset
        

    def config(self) -> typing.Dict[str, any]:
        return self._dataset_config

    def avgTrainError(self) -> torch.Tensor:
        num_elem = len(self._training_data_points)
        if num_elem == 0:
            return None
        
        avg_error = torch.tensor(0, dtype=self._dtype, device=self._device)
        for key in self._training_data_points:
            alignment_deviation = self._data_points[key].alignment_deviation()
            if alignment_deviation:
                avg_error = avg_error +  alignment_deviation
            else:
                num_elem = num_elem - 1

        if num_elem > 0:
            avg_error = avg_error / num_elem
        return avg_error

    def avgTestError(self) -> torch.Tensor:
        num_elem = len(self._testing_data_points)
        if num_elem == 0:
            return None
        
        avg_error = torch.tensor(0, dtype=self._dtype, device=self._device)
        for key in self._testing_data_points:
            alignment_deviation = self._data_points[key].alignment_deviation()
            if alignment_deviation:
                avg_error = avg_error +  alignment_deviation
            else:
                num_elem = num_elem - 1

        if num_elem > 0:
            avg_error = avg_error / num_elem
        return avg_error

    def avgEvalError(self) -> torch.Tensor:
        num_elem = len(self._evaluation_data_points)
        if num_elem == 0:
            return None
        
        avg_error = torch.tensor(0, dtype=self._dtype, device=self._device)
        for key in self._evaluation_data_points:
            alignment_deviation = self._data_points[key].alignment_deviation()
            if alignment_deviation:
                avg_error = avg_error +  alignment_deviation
            else:
                num_elem = num_elem - 1

        if num_elem > 0:
            avg_error = avg_error / num_elem
        return avg_error
    
    def avgTestDistance(self) -> torch.Tensor:
        num_elem = len(self._testing_data_points)
        if num_elem == 0:
            return None
        
        avg_dist = torch.tensor(0, dtype=self._dtype, device=self._device)
        for key in self._testing_data_points:
            dp_dist = self._data_points[key].hausdorff_distance()
            if dp_dist:
                avg_dist = avg_dist + dp_dist
            else:
                num_elem = num_elem - 1

        if num_elem > 0:
            avg_dist = avg_dist / num_elem
        return avg_dist

    def avgEvalDistance(self) -> torch.Tensor:
        num_elem = len(self._evaluation_data_points)
        if num_elem == 0:
            return None
        
        avg_dist = torch.tensor(0, dtype=self._dtype, device=self._device)
        for key in self._evaluation_data_points:
            dp_dist = self._data_points[key].hausdorff_distance()
            if dp_dist:
                avg_dist = avg_dist + dp_dist
            else:
                num_elem = num_elem - 1

        if num_elem > 0:
            avg_dist = avg_dist / num_elem
        return avg_dist

    def prepareDataForPlotting(self, plot_errors : bool = False, plot_hausdorff : bool = False):
        all_ax1 = []
        all_ax2 = []
        all_azim = []
        all_elev = []
        all_dates = []
        # initial_ax1 = []
        # initial_ax2 = []
        train_ax1 = []
        train_ax2 = []
        train_azim = []
        train_elev = []
        train_dates = []
        test_ax1 = []
        test_ax2 = []
        test_azim = []
        test_elev = []
        test_dates = []
        eval_ax1 = []
        eval_ax2 = []
        eval_azim = []
        eval_elev = []
        eval_dates = []
        
        # initial_err = []
        train_err = []
        train_hausdorff = []
        test_err = []
        test_hausdorff = []
        eval_err = []
        eval_hausdorff = []
        vmin = None
        vmin_hausdorff = 0
        vmax = None
        vmax_hausdorff = None
        max_ax = None

        for data_point in self._data_points.values():
            # if plot_errors and data_point.initial_error and data_point.initial_error > 0:
            #     initial_ax1.append(data_point.ax1_steps.detach().numpy())
            #     initial_ax2.append(data_point.ax2_steps.detach().numpy())
            #     initial_err.append(data_point.initial_error.detach().numpy())

            #     if not vmin or data_point.initial_error < vmin:
            #         vmin = data_point.initial_error
            #     if not vmax or data_point.initial_error > vmax:
            #         vmax = data_point.initial_error

            all_ax1.append(data_point.ax1_steps.detach().numpy())
            all_ax2.append(data_point.ax2_steps.detach().numpy())
            all_azim.append(torch.rad2deg(data_point.sourceAzim()).detach().numpy())
            all_elev.append(torch.rad2deg(data_point.sourceElev()).detach().numpy())
            all_dates.append(data_point.created_at)

            if not max_ax or data_point.ax1_steps.detach().numpy() > max_ax:
                max_ax = data_point.ax1_steps.detach().numpy()
            if not max_ax or data_point.ax2_steps.detach().numpy() > max_ax:
                max_ax = data_point.ax2_steps.detach().numpy()

        for key, data_point in self.trainingDataset().items():
            if plot_errors and not data_point.alignment_deviation():
                continue
            elif plot_errors:
                train_err.append(data_point.alignment_deviation().detach().numpy())
                
                if not vmin or data_point.alignment_deviation().detach().numpy() < vmin:
                    vmin = data_point.alignment_deviation().detach().numpy()
                if not vmax or data_point.alignment_deviation().detach().numpy() > vmax:
                    vmax = data_point.alignment_deviation().detach().numpy()

            train_ax1.append(data_point.ax1_steps.detach().numpy())
            train_ax2.append(data_point.ax2_steps.detach().numpy())
            train_azim.append(torch.rad2deg(data_point.sourceAzim()).detach().numpy())
            train_elev.append(torch.rad2deg(data_point.sourceElev()).detach().numpy())
            train_dates.append(data_point.created_at)
            train_hausdorff.append(0)

            if not max_ax or data_point.ax1_steps.detach().numpy() > max_ax:
                max_ax = data_point.ax1_steps.detach().numpy()
            if not max_ax or data_point.ax2_steps.detach().numpy() > max_ax:
                max_ax = data_point.ax2_steps.detach().numpy()

        for key, data_point in self.testingDataset().items():
            if plot_errors and not data_point.alignment_deviation():
                continue
            elif plot_hausdorff and not data_point.hausdorff_distance():
                continue
            
            if plot_errors:
                test_err.append(data_point.alignment_deviation().detach().numpy())
                if not vmin or data_point.alignment_deviation().detach().numpy() < vmin:
                    vmin = data_point.alignment_deviation().detach().numpy()
                if not vmax or data_point.alignment_deviation().detach().numpy() > vmax:
                    vmax = data_point.alignment_deviation().detach().numpy()

            if plot_hausdorff:
                test_hausdorff.append(data_point.hausdorff_distance().detach().numpy())
                if not vmin_hausdorff or data_point.hausdorff_distance().detach().numpy() < vmin_hausdorff:
                    vmin_hausdorff = data_point.hausdorff_distance().detach().numpy()
                if not vmax_hausdorff or data_point.hausdorff_distance().detach().numpy() > vmax_hausdorff:
                    vmax_hausdorff = data_point.hausdorff_distance().detach().numpy()

            test_ax1.append(data_point.ax1_steps.detach().numpy())
            test_ax2.append(data_point.ax2_steps.detach().numpy())
            test_azim.append(torch.rad2deg(data_point.sourceAzim()).detach().numpy())
            test_elev.append(torch.rad2deg(data_point.sourceElev()).detach().numpy())
            test_dates.append(data_point.created_at)

            if not max_ax or data_point.ax1_steps.detach().numpy() > max_ax:
                max_ax = data_point.ax1_steps.detach().numpy()
            if not max_ax or data_point.ax2_steps.detach().numpy() > max_ax:
                max_ax = data_point.ax2_steps.detach().numpy()

        for key, data_point in self.evaluationDataset().items():
            if plot_errors and not data_point.alignment_deviation():
                continue
            elif plot_hausdorff and not data_point.hausdorff_distance():
                continue

            if plot_errors:
                eval_err.append(data_point.alignment_deviation().detach().numpy())
                if not vmin or data_point.alignment_deviation().detach().numpy() < vmin:
                    vmin = data_point.alignment_deviation().detach().numpy()
                if not vmax or data_point.alignment_deviation().detach().numpy() > vmax:
                    vmax = data_point.alignment_deviation().detach().numpy()

            if plot_hausdorff:
                eval_hausdorff.append(data_point.hausdorff_distance().detach().numpy())

                if not vmin_hausdorff or data_point.hausdorff_distance().detach().numpy() < vmin_hausdorff:
                    vmin_hausdorff = data_point.hausdorff_distance().detach().numpy()
                if not vmax_hausdorff or data_point.hausdorff_distance().detach().numpy() > vmax_hausdorff:
                    vmax_hausdorff = data_point.hausdorff_distance().detach().numpy()

            eval_ax1.append(data_point.ax1_steps.detach().numpy())
            eval_ax2.append(data_point.ax2_steps.detach().numpy())
            eval_azim.append(torch.rad2deg(data_point.sourceAzim()).detach().numpy())
            eval_elev.append(torch.rad2deg(data_point.sourceElev()).detach().numpy())
            eval_dates.append(data_point.created_at)

            if not max_ax or data_point.ax1_steps.detach().numpy() > max_ax:
                max_ax = data_point.ax1_steps.detach().numpy()
            if not max_ax or data_point.ax2_steps.detach().numpy() > max_ax:
                max_ax = data_point.ax2_steps.detach().numpy()

        vmax = vmax if vmax else 0
        vmin = vmin if vmin else 0
        vmax_hausdorff = vmax_hausdorff if vmax_hausdorff else 0
        vmin_hausdorff = vmin_hausdorff if vmin_hausdorff else 0

        return [all_ax1, all_ax2, all_azim, all_elev, all_dates,
               train_ax1, train_ax2, train_azim, train_elev, train_dates,
               test_ax1, test_ax2, test_azim, test_elev, test_dates,
               eval_ax1, eval_ax2, eval_azim, eval_elev, eval_dates,
               train_err, test_err, eval_err, train_hausdorff, test_hausdorff, eval_hausdorff, vmin, vmax, vmin_hausdorff, vmax_hausdorff, max_ax] 

    def computeRegressionParameters(self,
                                        epochs: int = 1000,
                                        ) -> torch.Tensor:
        sample_data = torch.stack([dp.alignment_deviation().detach() for key, dp in self.evaluationDataset().items()])

        # ax + b = err
        a = torch.tensor(1, dtype=sample_data.dtype, device=self._device, requires_grad=True)
        b = torch.tensor(0, dtype=sample_data.dtype, device=self._device, requires_grad=True)

        optimizer = torch.optim.Adam([a,b], lr=0.01)
        loss_criterion = torch.nn.MSELoss(size_average=False)
        for epoch in range(epochs):
            optimizer.zero_grad()
            pred_list = []
            for key, dp in self.evaluationDataset().items():
                pred_list.append(a * dp.hausdorff_distance() + b)
            loss = loss_criterion(torch.stack(pred_list), sample_data)
            # print(str(epoch) + ': ' + str(loss.detach().numpy()))
            loss.backward()
            optimizer.step()

        return b,a

    def toDirectory(self, 
                    output_dir : str, 
                    config_name : typing.Optional[str] = None,
                    data_name : typing.Optional[str] = None,
                    ):
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        config_name = config_name if config_name else HeliostatDataset.default_config_name
        data_name = data_name if data_name else HeliostatDataset.default_data_name

        csv_reader = HeliostatDatasetCSV(dtype=self._dtype, device=self._device)
        for key in self._active_data_points:
            dp = self._data_points[key]
            csv_reader.addDataRow(data_point=dp)
        csv_reader.writeCSV(csv_path = os.path.join(output_dir, data_name))

        cfg_path = os.path.join(output_dir, config_name)
        
        with open(cfg_path, "w") as file:
            file.write(json.dumps(self._dataset_config))