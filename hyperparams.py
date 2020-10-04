################################
# Experiment Parameters        #
################################
epochs = 10000
batch_size = 16
save_step = 20000
image_step = 10000

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
num_layers = 3
model_dim = 512
num_embeddings = 512
d_ff = 2048
num_heads = 8
dropout = 0.1

# LayerNorm
layernorm_eps = 1e-6

# Encoder parameters
encoder_n_conv = 3

positional_encoding_max_len = 5000

# Decoder parameters
post_num_conv = 5

################################
# Optimization Hyperparameters #
################################
positive_stop_weight = 6.
