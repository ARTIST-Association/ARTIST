import torch
import typing

class AbstractInputEncoder:

    def __init__(self,
                 num_actuators : int,
                 num_env_states : int,
                 dtype : torch.dtype = torch.get_default_dtype(),
                 device : torch.device = torch.device('cpu'),
                ):
        self._num_actuators = num_actuators
        self._num_env_states = num_env_states
        self._dtype = dtype
        self._device = device

        # abstract class guard
        if type(self).__name__ == AbstractInputEncoder.__name__:
            raise Exception("Don't implement an abstract class!")

    def encodeInputs(self, normalized_actuator_steps: torch.Tensor, normalized_env_state: typing.Optional[typing.Dict[str,torch.Tensor]] = None) -> torch.Tensor :
        # abstract class guard
        raise Exception("Abstract method must be overridden!") 

    def numEncodedInputs(self):
        # abstract class guard
        raise Exception("Abstract method must be overridden!") 

    def parametersDict(self) -> typing.Dict[str,any]:
        # abstract class guard
        raise Exception("Abstract method must be overridden!") 

class PseudoFourierActuatorEncoder(AbstractInputEncoder):

    class Keys(typing.NamedTuple):
        num_actuators : str = 'num_actuators'
        num_env_states : str = 'num_env_states'
        encoding_degree : str = 'encoding_degree'
        encoding_factor : str = 'encoding_factor'
        input_pass_through : str = 'input_pass_through'

    keys = Keys()

    def __init__(self,
                 num_actuators : int,
                 num_env_states : int = 0,
                 encoding_degree : int = 1,
                 encoding_factor : int = 1,
                 input_pass_through : bool = True,
                 dtype : torch.dtype = torch.get_default_dtype(),
                 device : torch.device = torch.device('cpu'),
                ):
        self._encoding_degree = encoding_degree
        self._encoding_factor = encoding_factor
        self._input_pass_through = input_pass_through

        super().__init__(num_actuators=num_actuators,
                         num_env_states=num_env_states,
                         dtype=dtype,
                         device=device,
                            )

    def encodeInputs(self, normalized_actuator_steps: torch.Tensor, normalized_env_state: typing.Optional[typing.Dict[str,torch.Tensor]] = None) -> torch.Tensor :
        encoded_inputs_list = []
        if self._input_pass_through:
            for i in range(len(normalized_actuator_steps)):
                encoded_inputs_list.append(normalized_actuator_steps[i])
        
        for deg in range(self._encoding_degree):
            for i in range(len(normalized_actuator_steps)):
                encoded_inputs_list.append(normalized_actuator_steps[i] / ((deg + 1) ** self._encoding_factor) * torch.sin(2**(deg+1) * torch.pi * normalized_actuator_steps[i]))
                encoded_inputs_list.append(normalized_actuator_steps[i] / ((deg + 1) ** self._encoding_factor) * torch.cos(2**(deg+1) * torch.pi * normalized_actuator_steps[i]))

        if self._num_env_states > 0:
            for n_v in normalized_env_state.values():
                encoded_inputs_list.append(n_v)
        encoded_inputs = torch.tensor(encoded_inputs_list, dtype=self._dtype, device=self._device)
        return encoded_inputs

    def numEncodedInputs(self):
        # add number of env_states
        num_encoded_inputs = self._num_env_states

        # account for passed through actuator steps
        if self._input_pass_through:
            num_encoded_inputs = num_encoded_inputs + self._num_actuators

        # account for encoded actuator steps
        num_encoded_inputs = num_encoded_inputs + 2 * self._encoding_degree * self._num_actuators

        return num_encoded_inputs

    def parametersDict(self) -> typing.Dict[str,any]:
        dict = {}
        dict[PseudoFourierActuatorEncoder.keys.num_actuators] = self._num_actuators
        dict[PseudoFourierActuatorEncoder.keys.num_env_states] = self._num_env_states
        dict[PseudoFourierActuatorEncoder.keys.encoding_degree] = self._encoding_degree
        dict[PseudoFourierActuatorEncoder.keys.encoding_factor] = self._encoding_factor
        dict[PseudoFourierActuatorEncoder.keys.input_pass_through] = self._input_pass_through
        return dict