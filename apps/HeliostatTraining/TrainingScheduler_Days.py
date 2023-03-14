# system dependencies
import sys
import os
import typing
import torch
import datetime

# local dependencies
lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir, os.pardir, 'lib'))
sys.path.append(lib_dir)
from CSVToolsLib.HeliostatDatasetCSV import HeliOSDatasetCSV
import HeliostatTrainingLib.HeliostatDatasetBuilder as HDB
import HeliostatTrainingLib.HeliostatDataset as HD
import HeliostatTrainingLib.HeliostatTraining as HT
import HeliostatTrainingLib.AlignmentTrainer as AT
import HeliostatKinematicLib.AlignmentModel as AM
import HeliostatKinematicLib.AlignmentDisturbanceModel as ADM
import NeuralNetworksLib.InputEncoding as IE

def main(training_dir: typing.Optional[str] = None):
    torch.set_default_dtype(torch.float64)
    training_dir = training_dir if training_dir else '/Users/moritz/Projekte/Alignment Optimizer/data/Training Results' #os.path.abspath(os.path.join(__file__, os.pardir, 'TrainingBase'))

    builder = HDB.HeliostatDatasetBuilder(dtype=torch.float64)
    heliostat_list = ['AJ.23']
    file_path = '/Users/moritz/Projekte/Alignment Optimizer/data/calibdata.csv'
    position = torch.tensor([-57.2, 66.4, 88.729]) # AJ.23

    # heliostat_list = ['AM.35']
    # file_path = '/Users/moritz/Projekte/Alignment Optimizer/data/calibdata.csv'
    # position = torch.tensor([-4.4, 80.3, 88.735]) # AM.35

    # heliostat_list = ['AM.42']
    # file_path = '/Users/moritz/Projekte/Alignment Optimizer/data/calibdata.csv'
    # position = torch.tensor([26.4, 80.3, 88.751]) # AM.42

    # heliostat_list = ['AM.43']
    # file_path = '/Users/moritz/Projekte/Alignment Optimizer/data/calibdata.csv'
    # position = torch.tensor([30.8, 80.3, 88.753]) # AM.43

    # heliostat_list = ['BA.28']
    # file_path = '/Users/moritz/Projekte/Alignment Optimizer/data/calibdata_BA28.csv'
    # position = torch.tensor([-35.2, 171.4, 88.68751]) # BA.28
    

    # rigid body disturbances
    disturbance_list = [
        'position_azim',
        'position_elev',
        'position_rad',

        'joint_2_cosys_pivot_azim',
        'joint_2_cosys_pivot_elev',
        'joint_2_cosys_pivot_rad',

        'concentrator_cosys_pivot_azim',
        'concentrator_cosys_pivot_elev',
        'concentrator_cosys_pivot_rad',

        'joint_1_east_tilt',
        'joint_1_north_tilt',
        'joint_1_up_tilt',

        'joint_2_east_tilt',
        'joint_2_north_tilt',
        'joint_2_up_tilt',

        'concentrator_east_tilt',
        'concentrator_north_tilt',
        'concentrator_up_tilt',

        'actuator_1_increment',
        'actuator_2_increment',
    ]

    # nn disturbances
    # disturbance_list = [
    #     'concentrator_cosys_pivot_azim',
    #     'concentrator_cosys_pivot_elev',
    #     'concentrator_cosys_pivot_rad',

    #     'concentrator_east_tilt',
    #     'concentrator_north_tilt',
    #     'concentrator_up_tilt',

    #     # 'actuator_1_increment',
    #     # 'actuator_2_increment',
    # ]
    
    # alignment = AM.PointAlignmentModel(position=position, disturbance_model=disturbance_model, dtype=torch.float64)

    start_date = datetime.datetime(year=2022, month=1, day=1)

    training_manager = AT.AlignmentTrainingManager(output_dir = training_dir, dtype=torch.float64, create_plots=True)

    for i in range(300):

        disturbance_model = ADM.RigidBodyAlignmentDisturbanceModel(disturbance_list = disturbance_list, dtype=torch.float64)

        # input_encoder = IE.PseudoFourierActuatorEncoder(num_actuators = 2,
        #                                             num_env_states = 4,
        #                                             encoding_degree = 2,
        #                                             )
        # disturbance_model = ADM.SNNAlignmentDisturbanceModel(disturbance_list=disturbance_list,
        #                                                     hidden_dim = 2, # best: 2
        #                                                     n_layers = 15,  # best: 15
        #                                                     input_encoder = input_encoder,
        #                                                     dtype=torch.float64,
        #                                                     )
        alignment = AM.HeliokonAlignmentModel(position=position, disturbance_model=disturbance_model, dtype=torch.float64)
        

        dataset = builder.buildByDate(data_points= file_path,
                                                    num_train_data = 60,
                                                    num_valid_data = 20,
                                                    num_test_data = 30,
                                                    start_i = 0,
                                                    csv_input_reader_type = HeliOSDatasetCSV,
                                                    heliostats_list = heliostat_list,
                                                    created_at_range = [start_date, None],
                                                    )

        if len(dataset.trainingDataset()) < 60 or len(dataset.testingDataset()) < 20 or len(dataset.evaluationDataset()) < 30:
            break

        min_created_at = None
        for dp in dataset.activeDataset().values():
            if not min_created_at or dp.created_at < min_created_at:
                min_created_at = dp.created_at

        if min_created_at == start_date:
            start_date = min_created_at + datetime.timedelta(days=1)
        else:
            start_date = min_created_at

        dataset = builder.buildWithMaximizedHausdorffDistance(
                                                        data_points=dataset.activeDataset(),
                                                        num_train_data = 60,
                                                        csv_input_reader_type = HeliOSDatasetCSV,
                                                        num_test_points = 20,
                                                        num_eval_points = 30,
                                                        heliostats_list = heliostat_list,
                                                        num_nearest_neighbors = 4,
                                                        )

        # dataset = builder.buildRandomlyFromNumbers(data_points=dataset.activeDataset(),
        #                                             num_train_data = 60,
        #                                             num_test_data = 20,
        #                                             num_eval_data = 30,
        #                                             csv_input_reader_type = HeliOSDatasetCSV,
        #                                             # perc_eval = 0.6,
        #                                             heliostats_list = heliostat_list,
        #                                             # created_at_range = [start_date,None],
        #                                             )
        

        training = HT.HeliostatTraining(dataset=dataset, alignment_model=alignment, name='Gauss AJ23 Date Sweep 2022 GM 4NN', dtype=torch.float64)
        training_manager.addTraining(training)

if __name__ == '__main__':
    main()