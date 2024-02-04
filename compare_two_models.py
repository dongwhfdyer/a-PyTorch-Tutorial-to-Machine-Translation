from aihwkit.nn.conversion import convert_to_analog_mapped
from aihwkit.nn import AnalogSequential
import torch
from torch import nn
import torch.nn.functional as F
import youtokentome
import math
from analog_utils import *
from aihwkit_model import a_Transformer, a_MultiHeadAttention, a_PositionWiseFCNetwork
from aihwkit_model import a_Encoder
from aihwkit_model import a_Decoder
from model import MultiHeadAttention, PositionWiseFCNetwork, Encoder, Decoder, Transformer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bpe_model = youtokentome.BPE(model="data/bpe.model")

checkpoint = torch.load("pretrained_transformer_checkpoint.pth.tar", map_location=device)

model_state = checkpoint['model']

vocab_size = model_state.vocab_size
positional_encoding = model_state.positional_encoding
d_inner = model_state.d_inner
n_layers = model_state.n_layers
dropout = model_state.dropout


# encoder = model_state.encoder
# decoder = model_state.decoder

def translate(source_sequence, beam_size=4, length_norm_coefficient=0.6):
    """
    Translates a source language sequence to the target language, with beam search decoding.

    :param source_sequence: the source language sequence, either a string or tensor of bpe-indices
    :param beam_size: beam size
    :param length_norm_coefficient: co-efficient for normalizing decoded sequences' scores by their lengths
    :return: the best hypothesis, and all candidate hypotheses
    """
    with torch.no_grad():
        # Beam size
        k = beam_size

        # Minimum number of hypotheses to complete
        n_completed_hypotheses = min(k, 10)

        # Vocab size
        vocab_size = bpe_model.vocab_size()

        # If the source sequence is a string, convert to a tensor of IDs
        if isinstance(source_sequence, str):
            encoder_sequences = bpe_model.encode(source_sequence,
                                                 output_type=youtokentome.OutputType.ID,
                                                 bos=False,
                                                 eos=False)
            encoder_sequences = torch.LongTensor(encoder_sequences).unsqueeze(0)  # (1, source_sequence_length)
        else:
            encoder_sequences = source_sequence
        encoder_sequences_input = encoder_sequences.to(device)  # (1, source_sequence_length)
        encoder_sequence_lengths = torch.LongTensor([encoder_sequences_input.size(1)]).to(device)  # (1)

        # Encode
        encoder_sequences = a_model(encoder_sequences_input, encoder_sequence_lengths)  # (1, source_sequence_length, d_model)
        # encoder_sequences = model(encoder_sequences_input, encoder_sequence_lengths)  # (1, source_sequence_length, d_model)

        # Our hypothesis to begin with is just <BOS>
        hypotheses = torch.LongTensor([[bpe_model.subword_to_id('<BOS>')]]).to(device)  # (1, 1)
        hypotheses_lengths = torch.LongTensor([hypotheses.size(1)]).to(device)  # (1)

        # Tensor to store hypotheses' scores; now it's just 0
        hypotheses_scores = torch.zeros(1).to(device)  # (1)

        # Lists to store completed hypotheses and their scores
        completed_hypotheses = list()
        completed_hypotheses_scores = list()

        # Start decoding
        step = 1

        # Assume "s" is the number of incomplete hypotheses currently in the bag; a number less than or equal to "k"
        # At this point, s is 1, because we only have 1 hypothesis to work with, i.e. "<BOS>"
        while True:
            s = hypotheses.size(0)
            decoder_sequences = a_decoder(decoder_sequences=hypotheses,

                                          decoder_sequence_lengths=hypotheses_lengths,
                                          encoder_sequences=encoder_sequences.repeat(s, 1, 1),
                                          encoder_sequence_lengths=encoder_sequence_lengths.repeat(
                                              s))  # (s, step, vocab_size)

            # Scores at this step
            scores = decoder_sequences[:, -1, :]  # (s, vocab_size)
            scores = F.log_softmax(scores, dim=-1)  # (s, vocab_size)

            # Add hypotheses' scores from last step to scores at this step to get scores for all possible new hypotheses
            scores = hypotheses_scores.unsqueeze(1) + scores  # (s, vocab_size)

            # Unroll and find top k scores, and their unrolled indices
            top_k_hypotheses_scores, unrolled_indices = scores.view(-1).topk(k, 0, True, True)  # (k)

            # Convert unrolled indices to actual indices of the scores tensor which yielded the best scores
            prev_word_indices = unrolled_indices // vocab_size  # (k)
            next_word_indices = unrolled_indices % vocab_size  # (k)

            # Construct the the new top k hypotheses from these indices
            top_k_hypotheses = torch.cat([hypotheses[prev_word_indices], next_word_indices.unsqueeze(1)],
                                         dim=1)  # (k, step + 1)

            # Which of these new hypotheses are complete (reached <EOS>)?
            complete = next_word_indices == bpe_model.subword_to_id('<EOS>')  # (k), bool

            # Set aside completed hypotheses and their scores normalized by their lengths
            # For the length normalization formula, see
            # "Google’s Neural Machine Translation System: Bridging the Gap between Human and Machine Translation"
            completed_hypotheses.extend(top_k_hypotheses[complete].tolist())
            norm = math.pow(((5 + step) / (5 + 1)), length_norm_coefficient)
            completed_hypotheses_scores.extend((top_k_hypotheses_scores[complete] / norm).tolist())

            # Stop if we have completed enough hypotheses
            if len(completed_hypotheses) >= n_completed_hypotheses:
                break

            # Else, continue with incomplete hypotheses
            hypotheses = top_k_hypotheses[~complete]  # (s, step + 1)
            hypotheses_scores = top_k_hypotheses_scores[~complete]  # (s)
            hypotheses_lengths = torch.LongTensor(hypotheses.size(0) * [hypotheses.size(1)]).to(device)  # (s)

            # Stop if things have been going on for too long
            if step > 100:
                break
            step += 1

        # If there is not a single completed hypothesis, use partial hypotheses
        if len(completed_hypotheses) == 0:
            completed_hypotheses = hypotheses.tolist()
            completed_hypotheses_scores = hypotheses_scores.tolist()

        # Decode the hypotheses
        all_hypotheses = list()
        for i, h in enumerate(bpe_model.decode(completed_hypotheses, ignore_ids=[0, 2, 3])):
            all_hypotheses.append({"hypothesis": h, "score": completed_hypotheses_scores[i]})

        # Find the best scoring completed hypothesis
        i = completed_hypotheses_scores.index(max(completed_hypotheses_scores))
        best_hypothesis = all_hypotheses[i]["hypothesis"]

        return best_hypothesis, all_hypotheses


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
    elif model_name == 'positionwisefcnetwork':
        a_model = a_PositionWiseFCNetwork(d_model=512,
                                          d_inner=d_inner,
                                          dropout=dropout).to(device)

        model = PositionWiseFCNetwork(d_model=512,
                                      d_inner=d_inner,
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

        model = Encoder(vocab_size=vocab_size,
                        positional_encoding=positional_encoding,
                        d_model=512,
                        n_heads=8,
                        d_queries=64,
                        d_values=64,
                        d_inner=d_inner,
                        n_layers=n_layers,
                        dropout=dropout).to(device)
        model.load_state_dict(model_state.encoder.state_dict())

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

        model = Decoder(vocab_size=vocab_size,
                        positional_encoding=positional_encoding,
                        d_model=512,
                        n_heads=8,
                        d_queries=64,
                        d_values=64,
                        d_inner=d_inner,
                        n_layers=n_layers,
                        dropout=dropout).to(device)
        model.load_state_dict(model_state.decoder.state_dict())

    elif model_name == 'transformer':
        a_model = a_Transformer(vocab_size=vocab_size,
                                positional_encoding=positional_encoding,
                                d_model=512,
                                n_heads=8,
                                d_queries=64,
                                d_values=64,
                                d_inner=d_inner,
                                n_layers=n_layers,
                                dropout=dropout).to(device)
        model = Transformer(vocab_size=vocab_size,
                            positional_encoding=positional_encoding,
                            d_model=512,
                            n_heads=8,
                            d_queries=64,
                            d_values=64,
                            d_inner=d_inner,
                            n_layers=n_layers,
                            dropout=dropout).to(device)
        model.load_state_dict(model_state.state_dict())



    else:
        raise ValueError('model_name is not correct')
    # log model initialization
    print("model_name: ", model_name)
    return model, a_model


def models_transfer_weights(model, a_model, model_name):
    print("begin to transfer model weights")

    model_dict = model.state_dict()
    if model_name == 'multiheadattention':
        # a_model.cast_queries.set_weights(model_dict['cast_queries.weight'], model_dict['cast_queries.bias'])
        # a_model.cast_keys_values.set_weights(model_dict['cast_keys_values.weight'], model_dict['cast_keys_values.bias'])
        # a_model.cast_output.set_weights(model_dict['cast_output.weight'], model_dict['cast_output.bias'])
        #
        # a_model.layer_norm.weight = nn.Parameter(model_dict['layer_norm.weight'])
        # a_model.layer_norm.bias = nn.Parameter(model_dict['layer_norm.bias'])
        # ---------kkuhn-block------------------------------ # use copy_weights function
        a_model.copy_weights(model)
        # ---------kkuhn-block------------------------------

        # #---------kkuhn-block------------------------------ # use set_weights function
        # a_model.set_weights(model_dict['cast_queries.weight'], model_dict['cast_queries.bias'],
        #                     model_dict['cast_keys_values.weight'], model_dict['cast_keys_values.bias'],
        #                     model_dict['cast_output.weight'], model_dict['cast_output.bias'],
        #                     model_dict['layer_norm.weight'], model_dict['layer_norm.bias'])
        # #---------kkuhn-block------------------------------

    elif model_name == 'positionwisefcnetwork':
        # a_model.fc1.set_weights(model_dict['fc1.weight'], model_dict['fc1.bias'])
        # a_model.fc2.set_weights(model_dict['fc2.weight'], model_dict['fc2.bias'])
        # a_model.layer_norm.weight = nn.Parameter(model_dict['layer_norm.weight'])
        # a_model.layer_norm.bias = nn.Parameter(model_dict['layer_norm.bias'])
        a_model.set_weights(model_dict['fc1.weight'], model_dict['fc1.bias'],
                            model_dict['fc2.weight'], model_dict['fc2.bias'],
                            model_dict['layer_norm.weight'], model_dict['layer_norm.bias'])
    elif model_name == 'encoder':
        # ---------kkuhn-block------------------------------ # set weights by custom function
        a_model.copy_weights(model)
        # ---------kkuhn-block------------------------------
        # #---------kkuhn-block------------------------------ # by analogsequential
        # a_model = AnalogSequential(convert_to_analog_mapped(model, rpu_config))
        # a_model.remap_analog_weights()
        # #---------kkuhn-block------------------------------
    elif model_name == 'decoder':
        #---------kkuhn-block------------------------------ #  set weights by custom function
        a_model.copy_weights(model)
        #---------kkuhn-block------------------------------
        # # ---------kkuhn-block------------------------------ #  by analogsequential
        # a_model = AnalogSequential(convert_to_analog_mapped(model, rpu_config))
        # a_model.remap_analog_weights()
        # # ---------kkuhn-block------------------------------
    elif model_name == 'transformer':
        a_model.copy_weights(model)
    else:
        raise ValueError('model_name is not correct')
    return a_model


def models_inference(model, a_model, model_name):
    model.eval()
    a_model.eval()
    if model_name == 'multiheadattention':

        query_sequences_sample = torch.rand(1, 12, 512).to(device)
        key_sequences_sample = torch.rand(1, 12, 512).to(device)
        key_value_sequence_lengths = torch.tensor([12]).to(device)
        input = (query_sequences_sample, key_sequences_sample, key_value_sequence_lengths)
    elif model_name == 'positionwisefcnetwork':
        input = (torch.rand(1, 12, 512).to(device))
    elif model_name == 'encoder':
        # input = (torch.randint(1, 20000, (1, 12)).to(device), torch.Tensor([12]).to(device))
        input = (torch.tensor([[4265, 4065, 3786, 4643, 3811, 19516, 3942, 4065, 3786, 20521,
                                3811, 17399]], device='cuda:0'), torch.tensor([12], device='cuda:0'))
    elif model_name == 'decoder':
        input = (torch.tensor([[2]], dtype=torch.long, device=device), torch.tensor([1], dtype=torch.long, device=device), torch.rand(1, 12, 512, device=device), torch.tensor([12], dtype=torch.long, device=device))

        # a_input = {
        #     "decoder_sequences": input[0],
        #     "decoder_sequence_lengths": input[1],
        #     "encoder_sequences": input[2],
        #     "encoder_sequence_lengths": input[3]
        # }

    elif model_name == 'transformer':
        input = (torch.randint(low=0, high=2, size=(1, 12, 512), dtype=torch.long, device=device), torch.tensor([[2]], dtype=torch.long, device=device), torch.tensor([12], dtype=torch.long, device=device), torch.tensor([1], dtype=torch.long, device=device),)
    else:
        raise ValueError('model_name is not correct')
    output = model(*input)
    # a_model(a_input, decoder_sequence_lengths=input[0])
    # a_output = a_model(
    #     encoder_sequences=input[2],
    #     encoder_sequence_lengths=input[3]
    # )
    a_output = a_model(*input)
    # model_stt = model.state_dict()
    # a_model_stt = a_model.state_dict()
    # ---------kkuhn-block------------------------------
    print(translate("It was the best of times, it was the worst of times."))
    # ---------kkuhn-block------------------------------

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
