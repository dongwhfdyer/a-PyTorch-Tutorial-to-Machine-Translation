# transformer.encoder written in torch
en = Encoder(vocab_size=vocab_size,
             positional_encoding=positional_encoding,
             d_model=512,
             n_heads=8,
             d_queries=64,
             d_values=64,
             d_inner=d_inner,
             n_layers=n_layers,
             dropout=dropout).to(device)

# transformer.encoder written in aihwkit
a_en = a_Encoder(vocab_size=vocab_size,
                 positional_encoding=positional_encoding,
                 d_model=512,
                 n_heads=8,
                 d_queries=64,
                 d_values=64,
                 d_inner=d_inner,
                 n_layers=n_layers,
                 dropout=dropout).to(device)

# load weights from checkpoint for torch's transformer.encoder
# !!!! This loading process makes the inconsistency in the output of torch's transformer.encoder and aihwkit's transformer.encoder
#---------kkuhn-block------------------------------ # !!!!!!!!!
checkpoint = torch.load("averaged_transformer_checkpoint.pth.tar", map_location=device)
en.load_state_dict(checkpoint['model'].encoder.state_dict())
#---------kkuhn-block------------------------------


# transfer weights from torch's transformer.encoder to aihwkit's transformer.encoder
a_en.copy_weights(en)

# set torch's transformer.encoder and aihwkit's transformer.encoder to eval mode
en.eval()
a_en.eval()

# inference
input = (torch.tensor([[4265, 4065, 3786, 4643, 3811, 19516, 3942, 4065, 3786, 20521,
                        3811, 17399]], device='cuda:0'), torch.tensor([12], device='cuda:0'))

output = en(*input)
a_output = a_en(*input)

# compare the output of torch's transformer.encoder and aihwkit's transformer.encoder
output_norm = torch.norm(output, p='fro', dim=None, keepdim=False, out=None, dtype=None).numpy(force=True)
a_error = torch.norm(a_output - output, p='fro', dim=None, keepdim=False, out=None, dtype=None).numpy(force=True)
a_error_ave = a_error / output_norm
print("a_error_ave: ", a_error_ave)


def copy_weights(self, encoder):
    encoder_state = encoder.state_dict()
    self.embedding = encoder.embedding
    self.positional_encoding = encoder.positional_encoding
    self.layer_norm.weight = nn.Parameter(encoder_state['layer_norm.weight'])
    self.layer_norm.bias = nn.Parameter(encoder_state['layer_norm.bias'])
    for layer_ind in range(self.n_layers):
        # ---------kkuhn-block------------------------------ # not set value
        self.encoder_layers[layer_ind][0].set_weights(encoder_state['encoder_layers.' + str(layer_ind) + '.0.cast_queries.weight'],
                                                      encoder_state['encoder_layers.' + str(layer_ind) + '.0.cast_queries.bias'],
                                                      encoder_state['encoder_layers.' + str(layer_ind) + '.0.cast_keys_values.weight'],
                                                      encoder_state['encoder_layers.' + str(layer_ind) + '.0.cast_keys_values.bias'],
                                                      encoder_state['encoder_layers.' + str(layer_ind) + '.0.cast_output.weight'],
                                                      encoder_state['encoder_layers.' + str(layer_ind) + '.0.cast_output.bias'],
                                                      encoder_state['encoder_layers.' + str(layer_ind) + '.0.layer_norm.weight'],
                                                      encoder_state['encoder_layers.' + str(layer_ind) + '.0.layer_norm.bias'])

        self.encoder_layers[layer_ind][1].set_weights(encoder_state['encoder_layers.' + str(layer_ind) + '.1.fc1.weight'],
                                                      encoder_state['encoder_layers.' + str(layer_ind) + '.1.fc1.bias'],
                                                      encoder_state['encoder_layers.' + str(layer_ind) + '.1.fc2.weight'],
                                                      encoder_state['encoder_layers.' + str(layer_ind) + '.1.fc2.bias'],
                                                      encoder_state['encoder_layers.' + str(layer_ind) + '.1.layer_norm.weight'],
                                                      encoder_state['encoder_layers.' + str(layer_ind) + '.1.layer_norm.bias'])
