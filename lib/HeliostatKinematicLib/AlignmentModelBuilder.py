import torch

# system dependencies
import sys
import os
import typing
import torch
import pickle
import json

# local dependencies
module_dir = os.path.abspath(os.path.join(__file__, os.pardir))
sys.path.append(module_dir)
import AlignmentModel as AM
import AlignmentDisturbanceModel as ADM

lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(lib_dir)
import NeuralNetworksLib.InputEncoding as IE

class AlignmendModelBuilder:

    disturbance_model_category = [AM.AbstractAlignmentModelWithDisturbanceModel.__name__, 
                                  AM.TwoAxesAlignmentModel.__name__, 
                                  AM.HeliokonAlignmentModel.__name__,
                                  AM.PointAlignmentModel.__name__,
                                  ]

    json_indent = 4

    class Keys(typing.NamedTuple):
        model_type = 'model_type'
        model_parameters = 'model_parameters'
        disturbance_type = 'disturbance_type'
        disturbance_model_state = 'disturbance_model_state'
        disturbance_parameters = 'disturbance_parameters'
        disturbance_input_encoder_type = 'disturbance_input_encoder_type'
        disturbance_input_encoder_parameters = 'disturbance_input_encoder_parameters'
        disturbance_n_layers = 'disturbance_n_layers'
        disturbance_hidden_dim = 'disturbance_hidden_dim'
 
    keys = Keys()

    def __init__(self, 
                 dtype : torch.dtype = torch.get_default_dtype(),
                 device : torch.device = torch.device('cpu'),
                 ):
        self._dtype = dtype
        self._device = device

    def _unpackTorchDict(self, torch_dict: typing.Dict[str, torch.Tensor]) -> typing.Dict[str, any]:
        dict = {}
        for key, torch_item in torch_dict.items():
            dict[key] = torch_item.tolist() if isinstance(torch_item, torch.Tensor) else torch_item
        return dict

    def _packDict(self, dict: typing.Dict[str, any], requires_grad: bool = False) -> typing.Dict[str, torch.Tensor]:
        torch_dict = {}
        for key, item in dict.items():
            torch_dict[key] = item if (isinstance(item, str) or isinstance(item, bool)) else torch.tensor(item, dtype=self._dtype, device=self._device, requires_grad=requires_grad)
        return torch_dict

    def dictFromAlignmentModel(self, alignment_model: AM.AbstractAlignmentModel) -> typing.Dict[str, any]:
        alignment_model_type_name = type(alignment_model).__name__

        alignment_model_dict = {}
        alignment_model_dict[AlignmendModelBuilder.keys.model_type] = alignment_model_type_name
        alignment_model_dict[AlignmendModelBuilder.keys.model_parameters] = self._unpackTorchDict(torch_dict=alignment_model.parameterDict())

        alignment_model_dict[AlignmendModelBuilder.keys.disturbance_type] = None
        alignment_model_dict[AlignmendModelBuilder.keys.disturbance_model_state] = None
        alignment_model_dict[AlignmendModelBuilder.keys.disturbance_parameters] = None
        alignment_model_dict[AlignmendModelBuilder.keys.disturbance_input_encoder_type] = None
        alignment_model_dict[AlignmendModelBuilder.keys.disturbance_input_encoder_parameters] = None

        if alignment_model_type_name in AlignmendModelBuilder.disturbance_model_category:
            disturbance_type = type(alignment_model._disturbance_model).__name__
            alignment_model_dict[AlignmendModelBuilder.keys.disturbance_type] = disturbance_type
            alignment_model_dict[AlignmendModelBuilder.keys.disturbance_model_state] = self._unpackTorchDict(torch_dict=alignment_model._disturbance_model.modelStateDict())
            alignment_model_dict[AlignmendModelBuilder.keys.disturbance_parameters] = alignment_model._disturbance_model._disturbance_list
            
            if disturbance_type == ADM.SNNAlignmentDisturbanceModel.__name__:
                alignment_model_dict[AlignmendModelBuilder.keys.disturbance_n_layers] = alignment_model._disturbance_model._n_layers
                alignment_model_dict[AlignmendModelBuilder.keys.disturbance_hidden_dim] = alignment_model._disturbance_model._hidden_dim

            if alignment_model._disturbance_model._input_encoder:
                alignment_model_dict[AlignmendModelBuilder.keys.disturbance_input_encoder_type] = type(alignment_model._disturbance_model._input_encoder).__name__
                alignment_model_dict[AlignmendModelBuilder.keys.disturbance_input_encoder_parameters] = alignment_model._disturbance_model._input_encoder.parametersDict()

        return alignment_model_dict

    def alignmentModelFromDict(self, alignment_model_dict : typing.Dict[str, any]) -> AM.AbstractAlignmentModel:
        
        disturbance_model_type = alignment_model_dict[AlignmendModelBuilder.keys.disturbance_type]
        disturbance_model = None
        if disturbance_model_type:
            disturbance_input_encoder = None
            disturbance_input_encoder_type = alignment_model_dict[AlignmendModelBuilder.keys.disturbance_input_encoder_type]
            if disturbance_input_encoder_type:
                if disturbance_input_encoder_type == IE.PseudoFourierActuatorEncoder.__name__:
                    disturbance_input_encoder = IE.PseudoFourierActuatorEncoder(num_actuators=alignment_model_dict[AlignmendModelBuilder.keys.disturbance_input_encoder_parameters][IE.PseudoFourierActuatorEncoder.keys.num_actuators],
                                                                                num_env_states=alignment_model_dict[AlignmendModelBuilder.keys.disturbance_input_encoder_parameters][IE.PseudoFourierActuatorEncoder.keys.num_env_states],
                                                                                encoding_degree=alignment_model_dict[AlignmendModelBuilder.keys.disturbance_input_encoder_parameters][IE.PseudoFourierActuatorEncoder.keys.encoding_degree],
                                                                                encoding_factor=alignment_model_dict[AlignmendModelBuilder.keys.disturbance_input_encoder_parameters][IE.PseudoFourierActuatorEncoder.keys.encoding_factor],
                                                                                input_pass_through=alignment_model_dict[AlignmendModelBuilder.keys.disturbance_input_encoder_parameters][IE.PseudoFourierActuatorEncoder.keys.input_pass_through],
                                                                                dtype=self._dtype,
                                                                                device=self._device,
                                                                                )

            disturbance_list = alignment_model_dict[AlignmendModelBuilder.keys.disturbance_parameters]
            if disturbance_model_type == ADM.RigidBodyAlignmentDisturbanceModel.__name__:
                disturbance_model = ADM.RigidBodyAlignmentDisturbanceModel(disturbance_list=disturbance_list, 
                                                                            model_state=self._packDict(dict=alignment_model_dict[AlignmendModelBuilder.keys.disturbance_model_state], requires_grad=True), 
                                                                            input_encoder=disturbance_input_encoder,
                                                                            dtype=self._dtype, 
                                                                            device=self._device)
            elif disturbance_model_type == ADM.SNNAlignmentDisturbanceModel.__name__:
                disturbance_model = ADM.SNNAlignmentDisturbanceModel(disturbance_list=disturbance_list, 
                                                                        n_layers=alignment_model_dict[AlignmendModelBuilder.keys.disturbance_n_layers],
                                                                        hidden_dim=alignment_model_dict[AlignmendModelBuilder.keys.disturbance_hidden_dim],
                                                                        model_state=self._packDict(dict=alignment_model_dict[AlignmendModelBuilder.keys.disturbance_model_state], requires_grad=True), 
                                                                        input_encoder=disturbance_input_encoder,
                                                                        dtype=self._dtype, 
                                                                        device=self._device)

        alignment_model_type = alignment_model_dict[AlignmendModelBuilder.keys.model_type]
        alignment_model = None
        if alignment_model_type == AM.HeliokonAlignmentModel.__name__:
            parameter_dict = alignment_model_dict[AlignmendModelBuilder.keys.model_parameters]
            parameter_dict = self._packDict(dict=parameter_dict)
            alignment_model = AM.HeliokonAlignmentModel(disturbance_model=disturbance_model, 
                                                        parameter_dict=parameter_dict, 
                                                        dtype=self._dtype, device=self._device)
        elif alignment_model_type == AM.PointAlignmentModel.__name__:
            parameter_dict = alignment_model_dict[AlignmendModelBuilder.keys.model_parameters]
            parameter_dict = self._packDict(dict=parameter_dict)
            alignment_model = AM.PointAlignmentModel(disturbance_model=disturbance_model, 
                                                        parameter_dict=parameter_dict, 
                                                        dtype=self._dtype, device=self._device)
        return alignment_model

    def saveAlignmentModelDictToJSON(self, alignment_model: AM.AbstractAlignmentModel, save_path : str):
        alignment_model_dict = self.dictFromAlignmentModel(alignment_model=alignment_model)
        alignment_model_json = json.dumps(alignment_model_dict, indent=AlignmendModelBuilder.json_indent)
        with open(save_path, 'w') as file:
            file.write(alignment_model_json)

    def loadAligmentModelDictFromJSON(self, json_path: str) -> typing.Dict[str, any]:
        with open(json_path) as file:
            return json.load(file)