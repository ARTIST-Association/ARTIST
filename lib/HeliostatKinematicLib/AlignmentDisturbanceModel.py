# system dependencies
import torch
import typing
import pickle
import sys
import os

# local dependencies
# module_dir = os.path.abspath(os.path.join(__file__, os.pardir))
# sys.path.append(module_dir)
lib_dir = os.path.abspath(os.path.join(__file__, os.pardir, os.pardir))
sys.path.append(lib_dir)
import NeuralNetworksLib.SNN as SNN
import NeuralNetworksLib.InputEncoding as IE

class AbstractAlignmentDisturbanceModel:

    def __init__(self, 
                 disturbance_list : typing.List[str], 
                 model_state : typing.Optional[typing.Dict[str, torch.Tensor]] = None,
                 input_encoder : typing.Optional[IE.AbstractInputEncoder] = None,
                 include_env_states : bool = True,
                 dtype : torch.dtype = torch.get_default_dtype(),
                 device : torch.device = torch.device('cpu'),
    ):
        self._disturbance_list = disturbance_list
        self._input_encoder = input_encoder
        self._include_env_states = include_env_states
        self._dtype = dtype
        self._device = device

        # load state
        if model_state:
            self.loadStateDict(model_state=model_state)

        # abstract class guard
        if type(self).__name__ == AbstractAlignmentDisturbanceModel.__name__:
            raise Exception("Don't implement an abstract class!")

    def summary(self) -> str:
        summary_str = "Disturbance Model:\n"
        return summary_str

    def disturbanceModelPath(self, dir_path: str) -> str:
        model_path = os.path.join(dir_path, "disturbances.model")
        return model_path

    def loadDisturbanceModel(self, dir_path: str):
        file_path = self.disturbanceModelPath(dir_path=dir_path)
        file = open(file_path, 'rb')
        self = pickle.load(file)
        file.close()
        return self

    def saveDisturbanceModel(self, dir_path: str):
        file_path = self.disturbanceModelPath(dir_path=dir_path)
        file = open(file_path, 'wb')
        pickle.dump(self, file)
        file.close()

    def encodeInputs(self, normalized_actuator_steps: torch.Tensor, normalized_env_state: typing.Optional[typing.Dict[str,torch.Tensor]] = None) -> torch.Tensor :
        if self._input_encoder:
            return self._input_encoder.encodeInputs(normalized_actuator_steps=normalized_actuator_steps, normalized_env_state=normalized_env_state)
        else:
            encoded_inputs_list = []
            for i in range(len(normalized_actuator_steps)):
                encoded_inputs_list.append(normalized_actuator_steps[i])

            if normalized_env_state:
                for key, value in normalized_env_state.items():
                    encoded_inputs_list.append(value)

            encoded_inputs = torch.tensor(encoded_inputs_list, dtype=self._dtype, device=self._device)
            return encoded_inputs

    def predictDisturbances(self, normalized_actuator_steps: torch.Tensor, normalized_env_state: typing.Optional[typing.Dict[str,torch.Tensor]] = None) -> typing.Dict[str, torch.Tensor]:
        # abstract class guard
        raise Exception("Abstract method must be overridden!") 

    def modelParameters(self) -> typing.List[torch.Tensor]:
        # abstract class guard
        raise Exception("Abstract method must be overridden!") 

    def modelStateDict(self) -> typing.Dict[str, torch.Tensor]:
        # abstract class guard
        raise Exception("Abstract method must be overridden!") 
    
    def loadStateDict(self, model_state : typing.Dict[str, torch.Tensor]):
        # abstract class guard
        raise Exception("Abstract method must be overridden!") 

class DisturbanceModelSingleton(AbstractAlignmentDisturbanceModel):
    _G_MODELS : typing.Dict[str, AbstractAlignmentDisturbanceModel] = {}
    G_DEFAULT_NEXT_MODEL_PREFIX = 'model_'
    
    def __init__(self, 
                 model: AbstractAlignmentDisturbanceModel,
                 model_name: typing.Optional[str] = None):

        self._model_name = model_name

        if not self._model_name:
            self._model_name = DisturbanceModelSingleton.G_DEFAULT_NEXT_MODEL_PREFIX + str(len(self._G_MODELS))

        if (not self._model_name) in DisturbanceModelSingleton._G_MODELS:
            DisturbanceModelSingleton._G_MODELS[self._model_name] = model

    def summary(self) -> str:
        return self._singleton().summary()

    def reset():
        DisturbanceModelSingleton._G_MODELS = {}

    def _singleton(self):
        return DisturbanceModelSingleton._G_MODELS[self._model_name]

    def predictDisturbances(self, normalized_actuator_steps: torch.Tensor, normalized_env_state: typing.Optional[typing.Dict[str,torch.Tensor]] = None) -> typing.Dict[str, torch.Tensor]:
        return self._singleton().predictDisturbances(normalized_actuator_steps=normalized_actuator_steps, normalized_env_state=normalized_env_state)

    def modelConfigString(self) -> str:
        return self._singleton().modelConfigString()

    def modelParameters(self) -> typing.List[torch.Tensor]:
        return self._singleton().modelParameters()

    def modelStateDict(self) -> typing.Dict[str, torch.Tensor]:
        return self._singleton().modelStateDict() 
    
    def loadStateDict(self, model_state : typing.Dict[str, torch.Tensor]):
        self._singleton().loadStateDict(model_state=model_state)

class RigidBodyAlignmentDisturbanceModel(AbstractAlignmentDisturbanceModel):

    def __init__(self,
                 disturbance_list : typing.List[str], 
                 model_state : typing.Optional[typing.Dict[str, torch.Tensor]] = None,
                 input_encoder : typing.Optional[IE.AbstractInputEncoder] = None,
                 randomize_initial_disturbances : bool = False,
                 initial_disturbance_range : typing.Optional[float] = None,
                 include_env_states : bool = True,
                 dtype : torch.dtype = torch.get_default_dtype(),
                 device : torch.device = torch.device('cpu'),
                ):
        super().__init__(disturbance_list=disturbance_list,
                         input_encoder=input_encoder,
                         model_state=model_state,
                         include_env_states=include_env_states,
                         dtype=dtype,
                         device=device
                        )

        initial_disturbance_range = initial_disturbance_range if initial_disturbance_range else 1.0

        if not model_state:
            self._disturbance_dict = {}
            for dist in self._disturbance_list:
                if randomize_initial_disturbances:
                    self._disturbance_dict[dist] = (torch.rand(1, dtype=self._dtype, device=self._device, requires_grad=True)[0] - 0.5) * 2.0 * initial_disturbance_range
                else:
                    self._disturbance_dict[dist] = torch.tensor(0, dtype=self._dtype, device=self._device, requires_grad=True)
        else:
            self._disturbance_dict = model_state

    def summary(self) -> str:
        summary_str = super().summary()
        summary_str = summary_str + "RigidBody"
        return summary_str

    def checkForNan(self) -> bool:
        for tp in self._disturbance_dict.values():
            if torch.isnan(tp):
                return True
        return False

    # override
    def predictDisturbances(self, normalized_actuator_steps: torch.Tensor, normalized_env_state: typing.Optional[typing.Dict[str,torch.Tensor]] = None) -> typing.Dict[str, torch.Tensor]:
        self.checkForNan()
        return self._disturbance_dict

    # override
    def modelParameters(self) -> typing.List[torch.Tensor]:
        return self._disturbance_dict

    # override
    def modelStateDict(self) -> typing.Dict[str, torch.Tensor]:
        return self._disturbance_dict 
    
    # override
    def loadStateDict(self, model_state : typing.Dict[str, torch.Tensor]):
        self._disturbance_dict = model_state

class NeuralNetworkAlignmentDisturbanceModel(AbstractAlignmentDisturbanceModel):
    def __init__(self,
                 disturbance_list : typing.List[str], 
                 model: typing.Optional[torch.nn.Module] = None,
                 model_state : typing.Optional[typing.Dict[str, torch.Tensor]] = None,
                 input_encoder : typing.Optional[IE.AbstractInputEncoder] = None,
                 num_inputs: typing.Optional[int] = None,
                 randomize_initial_disturbances : bool = False,
                 initial_disturbance_range : typing.Optional[float] = None,
                 include_env_states : bool = True,
                 output_factor : typing.Optional[torch.Tensor] = None,
                 dtype : torch.dtype = torch.get_default_dtype(),
                 device : torch.device = torch.device('cpu'),
                ):
        self._model = model
        self._in_dim = input_encoder.numEncodedInputs() if input_encoder else num_inputs
        if not self._in_dim:
            raise Exception("Either Input Encoder or Number of Inputs must be given!") 
        self._out_dim = len(disturbance_list)

        super().__init__(disturbance_list=disturbance_list,
                         model_state=model_state,
                         input_encoder=input_encoder,
                         include_env_states=include_env_states,
                         dtype=dtype,
                         device=device,
                    )

        self._output_factor = output_factor if output_factor else torch.tensor(1.0, dtype=self._dtype, device=self._device)

        # initialize model parameters
        if model_state:
            self.loadStateDict(model_state=model_state)
        elif randomize_initial_disturbances:
            raise Exception("TODO: Implement Method!") # TODO

    def summary(self) -> str:
        summary_str = super().summary()
        summary_str = summary_str + "NeuralNetwork"
        return summary_str

    def predictDisturbances(self, normalized_actuator_steps: torch.Tensor, normalized_env_state: typing.Optional[typing.Dict[str,torch.Tensor]] = None) -> typing.Dict[str, torch.Tensor]:
        # prediction
        encoded_inputs = self.encodeInputs(normalized_actuator_steps=normalized_actuator_steps, normalized_env_state=normalized_env_state)
        pred_tensor = self._model.forward(encoded_inputs) * self._output_factor

        # disturbance dict
        disturbance_dict = {}
        for i, key in enumerate(self._disturbance_list):
            disturbance_dict[key] = pred_tensor[i]
        return disturbance_dict

    def modelParameters(self) -> typing.List[torch.Tensor]:
        return self._model.parameters()

    def modelStateDict(self) -> typing.Dict[str, torch.Tensor]:
        return self._model.state_dict()
    
    def loadStateDict(self, model_state : typing.Dict[str, torch.Tensor]):
        self._model.load_state_dict(model_state)

class SNNAlignmentDisturbanceModel(NeuralNetworkAlignmentDisturbanceModel):
    def __init__(self,
                 disturbance_list : typing.List[str], 
                 hidden_dim: int,
                 n_layers: int,
                 dropout_prob: float = 0.0,
                 model_state : typing.Optional[typing.Dict[str, torch.Tensor]] = None,
                 input_encoder : typing.Optional[IE.AbstractInputEncoder] = None,
                 num_inputs: typing.Optional[int] = None,
                 randomize_initial_disturbances : bool = False,
                 initial_disturbance_range : typing.Optional[float] = None,
                 include_env_states : bool = True,
                 dtype : torch.dtype = torch.get_default_dtype(),
                 device : torch.device = torch.device('cpu'),
                ):

        in_dim = input_encoder.numEncodedInputs() if input_encoder else num_inputs
        if not in_dim:
            raise Exception("Either Input Encoder or Number of Inputs must be given!") 
        out_dim = len(disturbance_list)
        self._hidden_dim = hidden_dim
        self._n_layers = n_layers
        self._dropout_prob = dropout_prob

        model = SNN.SNN(in_dim=in_dim,
                        out_dim=out_dim,
                        hidden_dim=self._hidden_dim,
                        n_layers=self._n_layers,
                        dropout_prob=self._dropout_prob,
                        dtype=dtype,
                        device=device,
                        )

        super().__init__(disturbance_list=disturbance_list,
                        model=model,
                        model_state=model_state,
                        input_encoder=input_encoder,
                        num_inputs=in_dim,
                        randomize_initial_disturbances=randomize_initial_disturbances,
                        initial_disturbance_range=initial_disturbance_range,
                        include_env_states=include_env_states,
                        dtype=dtype,
                        device=device,
                        )

    def summary(self) -> str:
        summary_str = super().summary()
        summary_str = summary_str + "SNN"
        return summary_str