from aihwkit.nn import AnalogLinear, AnalogSequential
from aihwkit.simulator.configs import InferenceRPUConfig
from aihwkit.inference import PCMLikeNoiseModel
from aihwkit.nn.conversion import convert_to_analog_mapped
from aihwkit.simulator.presets import PresetIOParameters

from aihwkit.simulator.configs import (
    InferenceRPUConfig,
    WeightModifierType,
    WeightClipType,
    WeightNoiseType,
    BoundManagementType,
    NoiseManagementType,
    WeightClipParameter,
    WeightModifierParameter,
    MappingParameter,
)
import functools

# from aihwkit.simulator.rpu_base.devices import WeightNoiseType, BoundManagementType, NoiseManagementType
# from aihwkit.simulator.rpu_base.tiles import WeightClipParameter, WeightModifierParameter, WeightModifierType


# mapping = MappingParameter(max_input_size=512,
#                            max_output_size=512,
#                            digital_bias=True,
#                            weight_scaling_omega=1)
# rpu_config = InferenceRPUConfig(mapping=mapping)
#
# rpu_config.noise_model = PCMLikeNoiseModel(g_max=25.0, prog_noise_scale=10, read_noise_scale=0, drift_scale=0, prog_coeff=2.5)
# rpu_config.clip.type = WeightClipType.FIXED_VALUE
# rpu_config.clip.fixed_value = 1.0


def create_ideal_rpu_config(tile_size=512):
    """Create RPU Config with ideal conditions"""
    rpu_config = InferenceRPUConfig(
        mapping=MappingParameter(
            digital_bias=True,
            learn_out_scaling=True,
            weight_scaling_omega=1.0,
            out_scaling_columnwise=False,
            weight_scaling_columnwise=True,
            max_input_size=tile_size,
            max_output_size=0,
        ),
        forward=PresetIOParameters(is_perfect=True),
        noise_model=PCMLikeNoiseModel(
            prog_noise_scale=0.0,
            read_noise_scale=0.0,
            drift_scale=0.0,
        ),
        drift_compensation=None,
    )
    return rpu_config


class GlobalDriftCompensation:
    pass


def create_rpu_config(modifier_noise, tile_size=512, dac_res=256, adc_res=256):
    """Create RPU Config emulated typical PCM Device"""
    rpu_config = InferenceRPUConfig(
        clip=WeightClipParameter(type=WeightClipType.FIXED_VALUE, fixed_value=1.0),
        modifier=WeightModifierParameter(
            rel_to_actual_wmax=True,
            type=WeightModifierType.ADD_NORMAL,
            std_dev=modifier_noise,
        ),
        mapping=MappingParameter(
            digital_bias=True,
            learn_out_scaling=True,
            weight_scaling_omega=1.0,
            out_scaling_columnwise=True,
            weight_scaling_columnwise=True,
            max_input_size=tile_size,
            max_output_size=0,
        ),
        forward=PresetIOParameters(
            w_noise_type=WeightNoiseType.PCM_READ,
            w_noise=0.0175,
            inp_res=dac_res,
            out_res=adc_res,
            out_bound=10.0,
            out_noise=0.04,
            bound_management=BoundManagementType.ITERATIVE,
            noise_management=NoiseManagementType.ABS_MAX,
        ),
        noise_model=PCMLikeNoiseModel(),
        drift_compensation=GlobalDriftCompensation(),
    )
    return rpu_config


rpu_config = create_rpu_config(0.1)
# rpu_config = create_ideal_rpu_config()


# AnalogLinear_ = functools.partial(AnalogLinear, bias=False, rpu_config=rpu_config)
AnalogLinear_ = functools.partial(AnalogLinear, rpu_config=rpu_config)
# AnalogLinear_2 = functools.partial(AnalogLinear, bias=True, rpu_config=rpu_config) # todo. In encoder, bias is False, in decoder, bias is True
