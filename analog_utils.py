from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.inference import PCMLikeNoiseModel
from aihwkit.simulator.configs.utils import WeightClipType, MappingParameter
from aihwkit.nn.conversion import convert_to_analog_mapped
import functools

mapping = MappingParameter(max_input_size=512,
                           max_output_size=512,
                           digital_bias=True,
                           weight_scaling_omega=1)
rpu_config = InferenceRPUConfig(mapping=mapping)

rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0, prog_noise_scale=10, read_noise_scale=0, drift_scale=0, prog_coeff=2.5)
rpu_config.clip.type = WeightClipType.FIXED_VALUE
rpu_config.clip.fixed_value = 1.0

AnalogLinear_ = functools.partial(AnalogLinear, bias=False, rpu_config=rpu_config)