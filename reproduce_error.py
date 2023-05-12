import torch
import youtokentome
from aihwkit_model import a_Encoder
from model import Encoder

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bpe_model = youtokentome.BPE(model="data/bpe.model")

checkpoint = torch.load("averaged_transformer_checkpoint.pth.tar", map_location=device)

model_state = checkpoint['model']

vocab_size = model_state.vocab_size
positional_encoding = model_state.positional_encoding
d_inner = model_state.d_inner
n_layers = model_state.n_layers
dropout = model_state.dropout


def models_init(model_name):
    a_model = a_Encoder(vocab_size=vocab_size,
                        positional_encoding=positional_encoding,
                        d_model=512,
                        n_heads=8,
                        d_queries=64,
                        d_values=64,
                        d_inner=d_inner,
                        n_layers=n_layers,
                        dropout=dropout).to(device)

    model = Encoder(vocab_size=vocab_size,
                    positional_encoding=positional_encoding,
                    d_model=512,
                    n_heads=8,
                    d_queries=64,
                    d_values=64,
                    d_inner=d_inner,
                    n_layers=n_layers,
                    dropout=dropout).to(device)
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    #---------kkuhn-block------------------------------  # by uncommenting this line, you can reproduce the error.
    model.load_state_dict(model_state.encoder.state_dict())
    #---------kkuhn-block------------------------------
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    return model, a_model


def models_transfer_weights(model, a_model, model_name):
    print("begin to transfer model weights")
    model_dict = model.state_dict()
    a_model.copy_weights(model)
    return a_model


def models_inference(model, a_model, model_name):
    model.eval()
    a_model.eval()
    input = (torch.tensor([[4265, 4065, 3786, 4643, 3811, 19516, 3942, 4065, 3786, 20521,
                            3811, 17399]], device='cuda:0'), torch.tensor([12], device='cuda:0'))
    output = model(*input)
    a_output = a_model(*input)

    return output, a_output


def calc_norm(output, a_output):
    output_norm = torch.norm(output, p='fro', dim=None, keepdim=False, out=None, dtype=None).numpy(force=True)
    a_error = torch.norm(a_output - output, p='fro', dim=None, keepdim=False, out=None, dtype=None).numpy(force=True)
    a_error_ave = a_error / output_norm
    return a_error_ave


if __name__ == '__main__':
    # MODEL_NAME = 'positionwisefcnetwork'
    # MODEL_NAME = 'multiheadattention'
    MODEL_NAME = 'encoder'
    # MODEL_NAME = 'decoder'
    # MODEL_NAME = 'transformer'

    model, a_model = models_init(MODEL_NAME)
    a_model = models_transfer_weights(model, a_model, MODEL_NAME)
    output, a_output = models_inference(model, a_model, MODEL_NAME)
    a_error_ave = calc_norm(output, a_output)
    print("a_error_ave: ", a_error_ave)
