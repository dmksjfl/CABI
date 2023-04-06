import torch

task = "pen-human-v0"
seed = 45

device = 'cuda' if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None
max_action = None

dynamics_path = None

log_dir = './logs/'

hidden_layer_size = 400
hidden_layers = 2
transition_layers = 4

transition_init_num = 7
transition_select_num = 5

real_data_ratio = 0.7
forward_train_epoch = 100
# 80 for adroit tasks
backward_train_epoch = 200
# 160 for adroit tasks

vae_train_epoch = 200
top_k = 0.2

transition_batch_size = 256
data_collection_per_epoch = 60e3
buffer_size = 2e6
steps_per_epoch = 1000
model_epoch = 200

uncertainty_mode = 'disagreement'
rew_penalty = 3.0
with_cnt = False

transition_lr = 1e-3
horizon = 3
