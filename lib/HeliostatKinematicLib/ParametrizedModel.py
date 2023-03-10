import torch
import typing

class AbstractParametrizedModel:
    class Keys(typing.NamedTuple):
        pass
    keys = Keys()

    class DistKeys(typing.NamedTuple):
        pass
    dist_keys = DistKeys()

    def __init__(self, 
                # parametrization
                parameter_dict: typing.Dict[str, torch.Tensor] = {},
                disturbance_dict: typing.Dict[str, torch.Tensor] = {},

                # disturbance factors
                dist_factor_rot: typing.Optional[torch.Tensor] = None,
                dist_factor_len: typing.Optional[torch.Tensor] = None,
                dist_factor_perc: typing.Optional[torch.Tensor] = None,

                # pytorch config
                device: torch.device = torch.device('cpu'),
                dtype: torch.dtype = torch.get_default_dtype(),
                ):

        # pytorch config
        self._device = device
        self._dtype = dtype

        # parametrization
        self._parameter_dict = parameter_dict
        self._disturbance_dict = disturbance_dict

        # disturbance factors
        # self._dist_factor_rot = dist_factor_rot if dist_factor_rot else torch.pi * torch.tensor(1.0 / 1000.0, dtype=self._dtype, device=self._device, requires_grad=False)
        # self._dist_factor_len = dist_factor_len if dist_factor_len else torch.tensor(1.0 / 10.0, dtype=self._dtype, device=self._device, requires_grad=False)
        # self._dist_factor_perc = dist_factor_perc if dist_factor_perc else torch.tensor(1.0 / 100.0, dtype=self._dtype, device=self._device, requires_grad=False)
        self._dist_factor_rot = dist_factor_rot if dist_factor_rot else torch.pi * torch.tensor(10.0 / 1000.0, dtype=self._dtype, device=self._device, requires_grad=False)
        self._dist_factor_len = dist_factor_len if dist_factor_len else torch.tensor(1.0, dtype=self._dtype, device=self._device, requires_grad=False)
        self._dist_factor_perc = dist_factor_perc if dist_factor_perc else torch.tensor(1.0 / 100.0, dtype=self._dtype, device=self._device, requires_grad=False)

        # abstract class guard
        if type(self).__name__ == AbstractParametrizedModel.__name__:
                raise Exception("Don't implement an abstract class!")

    def setParameterDict(self, parameter_dict: typing.Dict[str, torch.Tensor]):
        self.parameter_dict = parameter_dict

    def parameterDict(self) -> typing.Dict[str, torch.Tensor]:
        return self._parameter_dict

    def parameterList(self) -> typing.List[torch.Tensor]:
        return self._parameter_dict.values()

    def setDisturbanceDict(self, disturbance_dict: typing.Dict[str, torch.Tensor]):
        self._disturbance_dict = disturbance_dict

    def disturbanceDict(self) -> typing.Dict[str, torch.Tensor]:
        return self._disturbance_dict

    def disturbanceList(self) -> typing.List[torch.Tensor]:
        return self._disturbance_dict.values()

    def _cathesianFromPolarDist(self, polar_dist: torch.Tensor):
        # polar[0] = azimuth: 0° -> due north [0,1,0], 90° -> due east [1, 0, 0], -90° -> due west [-1, 0, 0]
        # polar[1] = elevation: 0° -> due north [0, 1, 0], 90° -> due up [0, 0, 1]
        # polar[2] = radius

        az = polar_dist[0] #* self._dist_factor_rot
        el = polar_dist[1] #* self._dist_factor_rot
        r = polar_dist[2] * self._dist_factor_len

        carth = torch.stack(
            [r * torch.sin(az) * torch.cos(el), r * torch.cos(az) * torch.cos(el), r * torch.sin(el)]
        )
        
        return carth

    def _rotParam(self, 
                    parameter_key : typing.Optional[str], 
                    disturbance_key : typing.Optional[str]
                    ) -> torch.Tensor:
        # param
        param = torch.tensor(0, dtype = self._dtype, device=self._device, requires_grad=False)
        if parameter_key and parameter_key in self._parameter_dict:
            param = self._parameter_dict[parameter_key]

        # disturbance
        dist = torch.tensor(0, dtype = self._dtype, device=self._device, requires_grad=False)
        if disturbance_key and disturbance_key in self._disturbance_dict:
            dist = self._disturbance_dict[disturbance_key]

        param = param + dist * self._dist_factor_rot
        return param

    def _percParam(self,
                    parameter_key : typing.Optional[str], 
                    disturbance_key : typing.Optional[str]
                    ) -> torch.Tensor:
        # param
        param = torch.tensor(0, dtype = self._dtype, device=self._device, requires_grad=False)
        if parameter_key and parameter_key in self._parameter_dict:
            param = self._parameter_dict[parameter_key]

        # disturbance
        dist = torch.tensor(0, dtype = self._dtype, device=self._device, requires_grad=False)
        if disturbance_key and disturbance_key in self._disturbance_dict:
            dist = self._disturbance_dict[disturbance_key]

        param = param * (1.0 + dist * self._dist_factor_perc)
        return param

    def _boolParam(self, parameter_key : typing.Optional[str]) -> bool:
        if parameter_key in self._parameter_dict:
            param = self._parameter_dict[parameter_key]
        else:
            return False
        return param

    def _tensorParam(self, parameter_key : typing.Optional[str]) -> typing.Optional[torch.Tensor]:
        if parameter_key in self._parameter_dict:
            param = self._parameter_dict[parameter_key]
        else:
            return None
        return param

    def _parametrizedVector(self, 
                            parameter_keys : typing.List[typing.Optional[str]],
                            disturbance_keys : typing.List[typing.Optional[str]],
                            ) -> torch.Tensor:
        # vector
        vec = torch.tensor([0,0,0], dtype=self._dtype, device=self._device, requires_grad=False)
        
        if not None in parameter_keys and all(key in self._parameter_dict for key in parameter_keys):
            vec = torch.stack(
                    [self._parameter_dict[parameter_keys[0]], 
                    self._parameter_dict[parameter_keys[1]],
                    self._parameter_dict[parameter_keys[2]]
                    ])

        # disturbance
        vec_dist = torch.tensor([0,0,0], dtype=self._dtype, device=self._device, requires_grad=False)

        if not None in disturbance_keys and all(key in self._disturbance_dict for key in disturbance_keys):
            polar_dist = torch.stack(
                    [self._disturbance_dict[disturbance_keys[0]], 
                    self._disturbance_dict[disturbance_keys[1]],
                    self._disturbance_dict[disturbance_keys[2]]
                    ])
            vec_dist = self._cathesianFromPolarDist(polar_dist=polar_dist)

        vec = vec + vec_dist
        return vec