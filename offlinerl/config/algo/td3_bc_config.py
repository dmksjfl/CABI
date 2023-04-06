import torch
from offlinerl.utils.exp import select_free_cuda

task = "pen-human-v0"

seed = 45

device = 'cuda' if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None
max_action = None

max_epoch = 200
steps_per_epoch = 5000
exploration_noise = 0.1
policy_noise = 0.2
noise_clip = 0.5
normalize = True
alpha = 2.5

## new add
log_dir = './logs/'
use_cabi = True
use_per = False

batch_size = 256
real_data_ratio = 0.7
hidden_layer_size = 256
layer_num = 2
actor_lr=3E-4
critic_lr=3E-4
discount = 0.99
soft_target_tau=5e-3