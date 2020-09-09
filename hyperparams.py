################################
# Experiment Parameters        #
################################
epochs = 1
batch_size = 16
save_step = 2000
image_step = 500

# Simple train
csv_dir_simple = '/home/keon/speech-datasets/LJSpeech-1.1/metadata.csv'
audio_dir_simple = '/home/keon/speech-datasets/LJSpeech-1.1/wavs/LJ001-{}.wav'
pad_token = 0
lr = 5e-6

# Loss proportion adjustment
loss_w_stop = 1e-4

################################
# Data Parameters              #
################################
prepared_data_dir = './prepared_data'
checkpoint_path = './checkpoint'
log_dir = './logs'
output_dir = './outputs'
cleaners = 'english_cleaners'
load_error_msg = "This type of dataset is not supported(yet)."

# Simple train
sample_vocab_size = 100
data_dir = '/home/ubuntu/Kyumin/Transformer/LJSpeech-1.1'
weight_dir = './weights'

################################
# Audio Parameters             #
################################
mel_channels = 80
hidden_dim = 256
n_fft = 2048
sr = 22050
preemphasis = 0.97
frame_shift = 0.0125
frame_length = 0.05
hop_length = int(sr*frame_shift)
win_length = int(sr*frame_length)
max_db = 100
ref_db = 20

################################
# Model Parameters             #
################################
# Make model
num_layers = 6
model_dim = 512
num_embeddings = 1024
d_ff = 2048
num_heads = 8
model_dropout = 0.1

# LayerNorm
layernorm_eps = 1e-6

# ConvNorm
convnorm_kernel_size = 1
convnorm_stride = 1
convnorm_padding = None
convnorm_dilation = 1
convnorm_bias = True
convnorm_w_init_gain = 'linear'

# Encoder parameters
encoder_n_conv = 3

position_ffn_dropout = 0.1
positional_encoding_max_len = 5000

# Decoder parameters
pre_dropout = 0.2
post_kernel_size = 5
post_num_conv = 5
post_dropout = 0.1

# Attention parameters
attention_dropout = 0.1

################################
# Optimization Hyperparameters #
################################
positive_stop_weight = 6.
