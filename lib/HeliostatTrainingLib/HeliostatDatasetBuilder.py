# system dependencies
import torch
import typing
import datetime
from yacs.config import CfgNode
import random
import sys
import os

# local dependencies
module_dir = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(module_dir)
from HeliostatDatapoint import HeliostatDataPoint
from HeliostatDataset import HeliostatDataset
# import HeliostatDatasetAnalyzer as ANALYZERS
from HausdorffMetric import HausdorffMetric as HAUSDORFF

lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(lib_dir)
from CSVToolsLib.HeliostatDatasetCSV import HeliostatDatasetCSV

class HeliostatDatasetBuilder:
    def __init__(self, 
                 dtype: torch.dtype = torch.get_default_dtype(),
                 device: torch.device = torch.device('cpu'),
                 ):
        self._dtype = dtype
        self._device = device

    def buildFromParameters(self,
                            data_points : typing.Union[str, typing.Dict[int, HeliostatDataPoint]],
                            csv_input_reader_type : typing.Callable = HeliostatDatasetCSV,
                            heliostats_list : typing.List[str] = [],
                            created_at_range : typing.List[datetime.datetime] = [None,None],

                            # training parameters
                            train_datapoints_list : typing.List[str] = [],
                            train_heliostats_list : typing.List[str] = [],
                            train_ax1_steps_range : typing.List[int] = [None,None],
                            train_ax2_steps_range : typing.List[int] = [None,None],
                            # train_aimpoint_range : typing.Optional[typing.List[torch.Tensor]] = None,
                            # train_to_source_range : typing.Optional[typing.List[torch.Tensor]] = None,
                            train_source_azim_deg_range : typing.List[float] = [None,None],
                            train_source_elev_deg_range : typing.List[float] = [None,None],
                            # train_source_pos_range : typing.Optional[typing.List[torch.Tensor]] = None,
                            train_created_at_range : typing.List[datetime.datetime] = [None,None],
                            # train_exclude_testing_data : bool = True,
                            # train_exclude_evaluation_data : bool = True,

                            # testing parameters
                            test_datapoints_list : typing.List[str] = [],
                            test_heliostats_list : typing.List[str] = [],
                            test_ax1_steps_range : typing.List[int] = [None,None],
                            test_ax2_steps_range : typing.List[int] = [None,None],
                            # test_aimpoint_range : typing.Optional[typing.List[torch.Tensor]] = None,
                            # test_to_source_range : typing.Optional[typing.List[torch.Tensor]] = None,
                            test_source_azim_deg_range : typing.List[float] = [None,None],
                            test_source_elev_deg_range : typing.List[float] = [None,None],
                            # test_source_pos_range : typing.Optional[typing.List[torch.Tensor]] = None,
                            test_created_at_range : typing.List[datetime.datetime] = [None,None],
                            # test_exclude_training_data : bool = True,
                            # test_exclude_evaluation_data : bool = True,

                            # eval parameters
                            eval_datapoints_list : typing.List[str] = [],
                            eval_heliostats_list : typing.List[str] = [],
                            eval_ax1_steps_range : typing.List[int] = [None,None],
                            eval_ax2_steps_range : typing.List[int] = [None,None],
                            # eval_aimpoint_range : typing.Optional[typing.List[torch.Tensor]] = None,
                            # eval_to_source_range : typing.Optional[typing.List[torch.Tensor]] = None,
                            eval_source_azim_deg_range : typing.List[float] = [None,None],
                            eval_source_elev_deg_range : typing.List[float] = [None,None],
                            # eval_source_pos_range : typing.Optional[typing.List[torch.Tensor]] = None,
                            eval_created_at_range : typing.List[datetime.datetime] = [None,None],
                            # eval_exclude_training_data : bool = True,
                            # eval_exclude_testing_data : bool = True,
                            ) -> HeliostatDataset:

        dataset_config = {}
        created_at_range[0] = created_at_range[0].strftime(HeliostatDataPoint._datetime_format) if created_at_range[0] else None
        created_at_range[1] = created_at_range[1].strftime(HeliostatDataPoint._datetime_format) if created_at_range[1] else None
        config_dict = {
            HeliostatDataset.config_keys.heliostats_list: heliostats_list,
            # HeliostatDataset.config_keys.data_point_list: [],
            HeliostatDataset.config_keys.date_range: created_at_range,
            # HeliostatDataset.config_keys.source_azim_range: [None, None],
            # HeliostatDataset.config_keys.source_elev_range: [None, None],
            # HeliostatDataset.config_keys.actuator_1_range: [None, None],
            # HeliostatDataset.config_keys.actuator_2_range: [None, None],
        }
        train_created_at_range[0] = train_created_at_range[0].strftime(HeliostatDataPoint._datetime_format) if train_created_at_range[0] else None
        train_created_at_range[1] = train_created_at_range[1].strftime(HeliostatDataPoint._datetime_format) if train_created_at_range[1] else None
        train_config_dict = {
            HeliostatDataset.config_keys.heliostats_list: train_heliostats_list,
            HeliostatDataset.config_keys.data_point_list: train_datapoints_list,
            HeliostatDataset.config_keys.date_range: train_created_at_range,
            HeliostatDataset.config_keys.source_azim_range: train_source_azim_deg_range,
            HeliostatDataset.config_keys.source_elev_range: train_source_elev_deg_range,
            HeliostatDataset.config_keys.actuator_1_range: train_ax1_steps_range,
            HeliostatDataset.config_keys.actuator_2_range: train_ax2_steps_range,
        }
        test_created_at_range[0] = test_created_at_range[0].strftime(HeliostatDataPoint._datetime_format) if test_created_at_range[0] else None
        test_created_at_range[1] = test_created_at_range[1].strftime(HeliostatDataPoint._datetime_format) if test_created_at_range[1] else None
        test_config_dict = {
            HeliostatDataset.config_keys.heliostats_list: test_heliostats_list,
            HeliostatDataset.config_keys.data_point_list: test_datapoints_list,
            HeliostatDataset.config_keys.date_range: test_created_at_range,
            HeliostatDataset.config_keys.source_azim_range: test_source_azim_deg_range,
            HeliostatDataset.config_keys.source_elev_range: test_source_elev_deg_range,
            HeliostatDataset.config_keys.actuator_1_range: test_ax1_steps_range,
            HeliostatDataset.config_keys.actuator_2_range: test_ax2_steps_range,
        }
        eval_created_at_range[0] = eval_created_at_range[0].strftime(HeliostatDataPoint._datetime_format) if eval_created_at_range[0] else None
        eval_created_at_range[1] = eval_created_at_range[1].strftime(HeliostatDataPoint._datetime_format) if eval_created_at_range[1] else None
        eval_config_dict = {
            HeliostatDataset.config_keys.heliostats_list: eval_heliostats_list,
            HeliostatDataset.config_keys.data_point_list: eval_datapoints_list,
            HeliostatDataset.config_keys.date_range: eval_created_at_range,
            HeliostatDataset.config_keys.source_azim_range: eval_source_azim_deg_range,
            HeliostatDataset.config_keys.source_elev_range: eval_source_elev_deg_range,
            HeliostatDataset.config_keys.actuator_1_range: eval_ax1_steps_range,
            HeliostatDataset.config_keys.actuator_2_range: eval_ax2_steps_range,
        }
        dataset_config[HeliostatDataset.config_keys.all] = config_dict
        dataset_config[HeliostatDataset.config_keys.training] = train_config_dict
        dataset_config[HeliostatDataset.config_keys.testing] = test_config_dict
        dataset_config[HeliostatDataset.config_keys.evaluation] = eval_config_dict
        
        dataset = HeliostatDataset(data_points=data_points, 
                                   csv_input_reader_type=csv_input_reader_type, 
                                   dataset_config=dataset_config,
                                   dtype=self._dtype,
                                   device=self._device
                                   )
        return dataset
    
    def buildRandomlyFromNumbers(self, 
                                    data_points : typing.Union[str, typing.Dict[int, HeliostatDataPoint]],
                                    num_train_data: int,
                                    num_test_data: int,
                                    num_eval_data: int,
                                    csv_input_reader_type : typing.Callable = HeliostatDatasetCSV,
                                    heliostats_list : typing.List[str] = [],
                                    created_at_range : typing.List[datetime.datetime] = [None,None],
                                    hausdorff_maximized_evaluation : bool = False
                              ) -> HeliostatDataset:
        dataset = self.buildFromParameters(data_points=data_points,
                                           csv_input_reader_type=csv_input_reader_type,
                                           heliostats_list=heliostats_list,
                                           created_at_range=created_at_range,
                                            )

        training_keys = list(dataset.trainingDataset().keys())
        config_dict = {
            HeliostatDataset.config_keys.data_point_list: training_keys.copy(),
        }
        num_data_points = num_test_data + num_eval_data

        if not hausdorff_maximized_evaluation:
            no_train_data_keys = random.sample(training_keys, num_data_points)
            training_keys = [tk for tk in training_keys if tk not in no_train_data_keys]

            evaluation_keys = random.sample(no_train_data_keys, num_eval_data)
            testing_keys = [ntk for ntk in no_train_data_keys if ntk not in evaluation_keys]

        if len(training_keys) > num_train_data:
            training_keys = random.sample(training_keys, num_train_data)

        dataset_config = {}
        train_config_dict = {
            HeliostatDataset.config_keys.data_point_list: training_keys,
        }
        test_config_dict = {
            HeliostatDataset.config_keys.data_point_list: testing_keys,
        }
        eval_config_dict = {
            HeliostatDataset.config_keys.data_point_list: evaluation_keys,
        }

        dataset_config[HeliostatDataset.config_keys.all] = config_dict
        dataset_config[HeliostatDataset.config_keys.training] = train_config_dict
        dataset_config[HeliostatDataset.config_keys.testing] = test_config_dict
        dataset_config[HeliostatDataset.config_keys.evaluation] = eval_config_dict

        dataset.setDataPoints(dataset_config=dataset_config)

        return dataset

    def buildRandomlyFromPercentage(self, 
                                    data_points : typing.Union[str, typing.Dict[int, HeliostatDataPoint]],
                                    num_train_data: int,
                                    csv_input_reader_type : typing.Callable = HeliostatDatasetCSV,
                                    perc_eval: float = 0.5,
                                    heliostats_list : typing.List[str] = [],
                                    created_at_range : typing.List[datetime.datetime] = [None,None],
                                    hausdorff_maximized_evaluation : bool = False
                              ) -> HeliostatDataset:
        dataset = self.buildFromParameters(data_points=data_points,
                                           csv_input_reader_type=csv_input_reader_type,
                                           heliostats_list=heliostats_list,
                                           created_at_range=created_at_range,
                                            )

        training_keys = list(dataset.trainingDataset().keys())
        config_dict = {
            HeliostatDataset.config_keys.data_point_list: training_keys.copy(),
        }
        num_data_points = len(training_keys) - num_train_data

        if not hausdorff_maximized_evaluation:
            no_train_data_keys = random.sample(training_keys, num_data_points)
            training_keys = [tk for tk in training_keys if tk not in no_train_data_keys]

            evaluation_keys = random.sample(no_train_data_keys, int(perc_eval * num_data_points))
            testing_keys = [ntk for ntk in no_train_data_keys if ntk not in evaluation_keys]

        dataset_config = {}
        train_config_dict = {
            HeliostatDataset.config_keys.data_point_list: training_keys,
        }
        test_config_dict = {
            HeliostatDataset.config_keys.data_point_list: testing_keys,
        }
        eval_config_dict = {
            HeliostatDataset.config_keys.data_point_list: evaluation_keys,
        }

        dataset_config[HeliostatDataset.config_keys.all] = config_dict
        dataset_config[HeliostatDataset.config_keys.training] = train_config_dict
        dataset_config[HeliostatDataset.config_keys.testing] = test_config_dict
        dataset_config[HeliostatDataset.config_keys.evaluation] = eval_config_dict

        dataset.setDataPoints(dataset_config=dataset_config)

        return dataset

    def buildWithMaximizedHausdorffDistance(self,
                                            data_points : typing.Union[str, typing.Dict[int, HeliostatDataPoint]],
                                            num_train_data: int,
                                            csv_input_reader_type : typing.Callable = HeliostatDatasetCSV,
                                            perc_eval: float = 0.5,
                                            num_test_points: typing.Optional[float] = None,
                                            num_eval_points: typing.Optional[float] = None,
                                            fill_with_closest: bool = False,
                                            heliostats_list : typing.List[str] = [],
                                            created_at_range : typing.List[datetime.datetime] = [None,None],
                                            dist_method : typing.Callable = HAUSDORFF.distanceByAngle,
                                            num_nearest_neighbors : int = 1,
                                            ) -> HeliostatDataset:

        ### Load Dataset ###
        dataset = self.buildFromParameters(data_points=data_points,
                                           csv_input_reader_type=csv_input_reader_type,
                                           heliostats_list=heliostats_list,
                                           created_at_range=created_at_range,
                                            )

        training_keys = list(dataset.trainingDataset().keys())
        config_dict = {
            HeliostatDataset.config_keys.data_point_list: training_keys.copy(),
        }

        ### Sort By Distance ###
        key_dists = {}
        for key in training_keys:
            dp = dataset._data_points[key]

            min_dist = dp.distanceToDataset(data_points = dataset._data_points,
                                            selected_data_points=[training_keys],
                                            dist_method=dist_method, 
                                            num_nearest_neighbors=num_nearest_neighbors,
                                            )
                    
            key_dists[min_dist] = key

        key_dists = dict(sorted(key_dists.items(), reverse=True))

        ### Data Splitting
        if not (num_test_points and num_eval_points):
            num_data_points = len(training_keys) - num_train_data
            num_eval_points = int(num_data_points * perc_eval + 0.5)
            num_test_points = int(num_data_points * (1.0 - perc_eval) + 0.5)

        evaluation_keys = []
        testing_keys = []

        key_list = [v for v in key_dists.values()]
        evaluation_keys = key_list[:num_eval_points]
        testing_keys = key_list[num_eval_points:(num_eval_points + num_test_points)]
        training_keys = key_list[(num_eval_points + num_test_points):]

        # while len(evaluation_keys) < num_eval_points or len(testing_keys) < num_test_points:
        #     total_max_dist = None
        #     total_max_key = None
        #     for key in training_keys:
        #         dp = dataset._data_points[key]
        #         min_dist = dp.distanceToDataset(data_points = dataset._data_points,
        #                                                  selected_data_points=[training_keys, testing_keys, evaluation_keys],
        #                                                  dist_method=dist_method, 
        #                                                  num_nearest_neighbors=num_nearest_neighbors,
        #                                                  )

        #         if not total_max_dist or min_dist > total_max_dist:
        #             total_max_dist = min_dist
        #             total_max_key = key

        #     if len(evaluation_keys) < num_eval_points:
        #         evaluation_keys.append(total_max_key)
        #         print(str(len(evaluation_keys)) + "/" + str(num_eval_points) + ': ' + str(total_max_key))
        #     elif len(testing_keys) < num_test_points:
        #         testing_keys.append(total_max_key)
        #         print(str(len(testing_keys)) + "/" + str(num_test_points) + ': ' + str(total_max_key))
        #     training_keys.remove(total_max_key)

        ### Fill With Closest ###
        if fill_with_closest:
            key_dists = {}
            for key in training_keys:
                dp = dataset._data_points[key]

                min_dist = dp.distanceToDataset(data_points = dataset._data_points,
                                                                    selected_data_points=[evaluation_keys],
                                                                    dist_method=dist_method, 
                                                                    num_nearest_neighbors=num_nearest_neighbors)
                            
                key_dists[min_dist] = key

            key_dists = dict(sorted(key_dists.items()))
            training_keys = [v for v in key_dists.values()]
        
        training_keys = training_keys[:num_train_data]

        ### Write Configs and Update Dataset ###
        dataset_config = {}
        train_config_dict = {
            HeliostatDataset.config_keys.data_point_list: training_keys,
        }
        test_config_dict = {
            HeliostatDataset.config_keys.data_point_list: testing_keys,
        }
        eval_config_dict = {
            HeliostatDataset.config_keys.data_point_list: evaluation_keys,
        }

        dataset_config[HeliostatDataset.config_keys.all] = config_dict
        dataset_config[HeliostatDataset.config_keys.training] = train_config_dict
        dataset_config[HeliostatDataset.config_keys.testing] = test_config_dict
        dataset_config[HeliostatDataset.config_keys.evaluation] = eval_config_dict

        dataset.setDataPoints(dataset_config=dataset_config)

        return dataset

    # def getDate(data_point : HeliostatDataPoint):
    #     return data_point.created_at

    def buildDistanceThreshold(self,
                                data_points : typing.Union[str, typing.Dict[int, HeliostatDataPoint]],
                                csv_input_reader_type : typing.Callable = HeliostatDatasetCSV,
                                heliostats_list : typing.List[str] = [],
                                created_at_range : typing.List[datetime.datetime] = [None,None],
                                dist_method : typing.Callable = HAUSDORFF.distanceByAngle,
                                threshold : float = 0.2,
                                ) -> HeliostatDataset:
        
        dataset = self.buildFromParameters(data_points=data_points,
                                           csv_input_reader_type=csv_input_reader_type,
                                           heliostats_list=heliostats_list,
                                           created_at_range=created_at_range,
                                            )
        training_keys = list(dataset.trainingDataset().keys())
        data_point_dates = {}

        for key in training_keys:
            dp = dataset._data_points[key]
            dp_date = datetime.date(year=dp.created_at.year, month=dp.created_at.month, day=dp.created_at.day)
            if dp_date in data_point_dates:
                data_point_dates[dp_date].append(key)
            else:
                data_point_dates[dp_date] = [key]

        data_point_dates = dict(sorted(data_point_dates.items()))
        new_training_keys = []
        validation_keys = []
        for dt in data_point_dates.keys():

            for key in data_point_dates[dt]:
                train_dist = dataset._data_points[key].distanceToDataset(data_points = dataset._data_points,
                                                                        selected_data_points=[new_training_keys],
                                                                        dist_method=dist_method, 
                                                                        num_nearest_neighbors=1,
                                                                        )
                valid_dist = dataset._data_points[key].distanceToDataset(data_points = dataset._data_points,
                                                                        selected_data_points=[validation_keys],
                                                                        dist_method=dist_method, 
                                                                        num_nearest_neighbors=1,
                                                                        )
                if len(new_training_keys) == 0 or train_dist > threshold:
                    new_training_keys.append(key)
                
                elif len(validation_keys) == 0 or valid_dist > threshold:
                    validation_keys.append(key)

        testing_keys = []
        for key in training_keys:
            if not key in new_training_keys and not key in validation_keys:
                testing_keys.append(key)

        dataset_config = {}
        train_config_dict = {
            HeliostatDataset.config_keys.data_point_list: new_training_keys,
        }
        test_config_dict = {
            HeliostatDataset.config_keys.data_point_list: validation_keys,
        }
        eval_config_dict = {
            HeliostatDataset.config_keys.data_point_list: testing_keys,
        }

        config_dict = {
            HeliostatDataset.config_keys.data_point_list: training_keys.copy(),
        }
        dataset_config[HeliostatDataset.config_keys.all] = config_dict
        dataset_config[HeliostatDataset.config_keys.training] = train_config_dict
        dataset_config[HeliostatDataset.config_keys.testing] = test_config_dict
        dataset_config[HeliostatDataset.config_keys.evaluation] = eval_config_dict

        dataset.setDataPoints(dataset_config=dataset_config)

        return dataset
        

    def buildByDate(self,
                    data_points : typing.Union[str, typing.Dict[int, HeliostatDataPoint]],
                    num_train_data: int,
                    num_valid_data: int,
                    num_test_data: int,
                    start_i: int,
                    csv_input_reader_type : typing.Callable = HeliostatDatasetCSV,
                    heliostats_list : typing.List[str] = [],
                    created_at_range : typing.List[datetime.datetime] = [None,None],
                    ) -> HeliostatDataset:

        dataset = self.buildFromParameters(data_points=data_points,
                                           csv_input_reader_type=csv_input_reader_type,
                                           heliostats_list=heliostats_list,
                                           created_at_range=created_at_range,
                                            )
# dict(sorted(people.items(), key=lambda item: item[1]))
        training_data_sorted = dict(sorted(dataset.trainingDataset().items(), key = lambda item : item[1].created_at))

        training_keys = list(dataset.trainingDataset().keys())
        config_dict = {
            HeliostatDataset.config_keys.data_point_list: training_keys.copy(),
        }

        training_keys = []
        validation_keys = []
        testing_keys = []

        for i, key in enumerate(list(training_data_sorted.keys())[start_i:]):
            # i = i + start_i
            if i < num_train_data:
                training_keys.append(key)
            elif i < num_train_data+num_valid_data:
                validation_keys.append(key)
            elif i < num_train_data+num_valid_data+num_test_data:
                testing_keys.append(key)

        dataset_config = {}
        train_config_dict = {
            HeliostatDataset.config_keys.data_point_list: training_keys,
        }
        test_config_dict = {
            HeliostatDataset.config_keys.data_point_list: validation_keys,
        }
        eval_config_dict = {
            HeliostatDataset.config_keys.data_point_list: testing_keys,
        }

        dataset_config[HeliostatDataset.config_keys.all] = config_dict
        dataset_config[HeliostatDataset.config_keys.training] = train_config_dict
        dataset_config[HeliostatDataset.config_keys.testing] = test_config_dict
        dataset_config[HeliostatDataset.config_keys.evaluation] = eval_config_dict

        dataset.setDataPoints(dataset_config=dataset_config)

        return dataset

# def buildDatasetExample():
#     builder = HeliostatDatasetBuilder(dataset=DATASETS.HeliOSHeliostatDataset(dist_method=HAUSDORFF.distanceBySteps))
#     heliostat_list = ['AM.42']
#     file_path = '/Users/Synhelion/Downloads/calibdata.csv'
#     # dataset = builder.buildRandomlyFromPercentage(file_path=file_path,
#     #                                               num_train_data=100,
#     #                                               perc_test=0.5,
#     #                                               heliostats_list=heliostat_list
#     #                                                 )
#     dataset = builder.buildWithMaximizedHausdorffDistance(file_path=file_path,
#                                                         num_train_data=100,
#                                                         perc_test=0.5,
#                                                         heliostats_list=heliostat_list
#                                                             )
#     # dataset = builder.buildByDate(file_path=file_path,
#     #                               num_train_data=50,
#     #                               perc_test=0.0,
#     #                               start_date=datetime.datetime(year=2022, month=1, day=1)
#     #                                 )
#     analyzer = ANALYZERS.HeliostatDatasetAnalyzer(dataset=dataset)
#     analyzer.plotDataDistributionOverAxes(plot_hausdorff=True, plot_epsilon_regions=True)
#     # analyzer.plotDataDistributionOverDates(plot_hausdorff=True)

# if __name__ == '__main__':
#     buildDatasetExample()