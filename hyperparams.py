################################
# Experiment Parameters        #
################################
# Simple train
batch_size = 30

################################
# Data Parameters             #
################################

################################
# Audio Parameters             #
################################
mel_channels = 80

################################
# Model Parameters             #
################################
# Make model
num_layers = 6
model_dim = 512
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
position_ffn_dropout = 0.1
positional_encoding_max_len = 5000

# Decoder parameters
post_num_conv = 5
post_dropout = 0.1

# Attention parameters
attention_dropout = 0.1

################################
# Optimization Hyperparameters #
################################
