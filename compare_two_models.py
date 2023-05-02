import torch
from torch import nn
import torch.nn.functional as F
import youtokentome
import math
from analog_utils import *
from aihwkit_model import a_Transformer, a_MultiHeadAttention
from aihwkit_model import a_Encoder
from aihwkit_model import a_Decoder
from model import MultiHeadAttention

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bpe_model = youtokentome.BPE(model="data/bpe.model")

checkpoint = torch.load("averaged_transformer_checkpoint.pth.tar", map_location=device)

model_state = checkpoint['model']

vocab_size = model_state.vocab_size
positional_encoding = model_state.positional_encoding
d_inner = model_state.d_inner
n_layers = model_state.n_layers
dropout = model_state.dropout
encoder = model_state.encoder
decoder = model_state.decoder


def models_init(model_name):
    if model_name == 'multiheadattention':
        a_model = a_MultiHeadAttention(d_model=512,
                                       n_heads=8,
                                       d_queries=64,
                                       d_values=64,
                                       dropout=dropout).to(device)

        model = MultiHeadAttention(d_model=512,
                                   n_heads=8,
                                   d_queries=64,
                                   d_values=64,
                                   dropout=dropout).to(device)
    elif model_name == 'encoder':
        a_model = a_Encoder(vocab_size=vocab_size,
                            positional_encoding=positional_encoding,
                            d_model=512,
                            n_heads=8,
                            d_queries=64,
                            d_values=64,
                            d_inner=d_inner,
                            n_layers=n_layers,
                            dropout=dropout).to(device)

        model = encoder.to(device)

    elif model_name == 'decoder':

        a_model = a_Decoder(vocab_size=vocab_size,
                            positional_encoding=positional_encoding,
                            d_model=512,
                            n_heads=8,
                            d_queries=64,
                            d_values=64,
                            d_inner=d_inner,
                            n_layers=n_layers,
                            dropout=dropout).to(device)

        model = decoder.to(device)

    elif model_name == 'transformer':
        a_model = a_Transformer(vocab_size=vocab_size,
                                positional_encoding=positional_encoding,
                                d_model=512,
                                n_heads=8,
                                d_queries=64,
                                d_values=64,
                                d_inner=d_inner,
                                n_layers=n_layers,
                                dropout=dropout)

        a_model.encoder = a_Encoder(vocab_size=vocab_size,
                                    positional_encoding=positional_encoding,
                                    d_model=512,
                                    n_heads=8,
                                    d_queries=64,
                                    d_values=64,
                                    d_inner=d_inner,
                                    n_layers=n_layers,
                                    dropout=dropout)

        a_model.decoder = a_Decoder(vocab_size=vocab_size,
                                    positional_encoding=positional_encoding,
                                    d_model=512,
                                    n_heads=8,
                                    d_queries=64,
                                    d_values=64,
                                    d_inner=d_inner,
                                    n_layers=n_layers, )
    else:
        raise ValueError('model_name is not correct')
    # log model initialization
    print("model_name: ", model_name)
    print("begin to transfer model weights")

    model_dict = model.state_dict()
    if model_name == 'multiheadattention':
        a_model.cast_queries.set_weights(model_dict['cast_queries.weight'], model_dict['cast_queries.bias'])
        a_model.cast_keys_values.set_weights(model_dict['cast_keys_values.weight'], model_dict['cast_keys_values.bias'])
        a_model.cast_output.set_weights(model_dict['cast_output.weight'], model_dict['cast_output.bias'])

        a_model.layer_norm.weight = nn.Parameter(model_dict['layer_norm.weight'])
        a_model.layer_norm.bias = nn.Parameter(model_dict['layer_norm.bias'])

    elif model_name == 'encoder':
        pass
    elif model_name == 'decoder':
        pass
    elif model_name == 'transformer':
        pass
    else:
        raise ValueError('model_name is not correct')
    model.eval()
    a_model.eval()
    return model, a_model


def models_inference(model, a_model, model_name):
    if model_name == 'multiheadattention':

        query_sequences_sample = torch.rand(1, 12, 512).to(device)
        key_sequences_sample = torch.rand(1, 12, 512).to(device)
        key_value_sequence_lengths = torch.tensor([12]).to(device)
        input = (query_sequences_sample, key_sequences_sample, key_value_sequence_lengths)

    elif model_name == 'encoder':
        pass
    elif model_name == 'decoder':
        pass
    elif model_name == 'transformer':
        pass
    else:
        raise ValueError('model_name is not correct')
    output = model(*input)
    a_output = a_model(*input)
    return output, a_output


def calc_norm(output, a_output):
    output_norm = torch.norm(output, p='fro', dim=None, keepdim=False, out=None, dtype=None).numpy(force=True)
    a_error = torch.norm(a_output - output, p='fro', dim=None, keepdim=False, out=None, dtype=None).numpy(force=True)
    a_error_ave = a_error / output_norm
    return a_error_ave


if __name__ == '__main__':
    MODEL_NAME = 'positionwisefcnetwork'
    # MODEL_NAME = 'encoder'
    # MODEL_NAME = 'decoder'
    # MODEL_NAME = 'transformer'

    model, a_model = models_init(MODEL_NAME)
    output, a_output = models_inference(model, a_model, MODEL_NAME)
    a_error_ave = calc_norm(output, a_output)
    print("a_error_ave: ", a_error_ave)
