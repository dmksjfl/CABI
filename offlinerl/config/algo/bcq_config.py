import torch
from offlinerl.utils.exp import select_free_cuda

task = "pen-human-v0"

seed = 42

device = 'cuda' if torch.cuda.is_available() else 'cpu'
obs_shape = None
act_shape = None
max_action = None

## new add
log_dir = './logs/'
use_cabi = True

vae_features = 750
vae_layers = 2
jitter_features = 400
jitter_layers = 2
value_features = 400
value_layers = 2
phi = 0.001
lam = 0.75

batch_size = 256
real_data_ratio = 0.9
steps_per_epoch = 5000
max_epoch = 200

use_per = False

# vae_lr = 3e-5
# jitter_lr = 3e-5
# critic_lr = 3e-4
vae_lr = 1e-3
jitter_lr = 3e-4
critic_lr = 3e-4
kl_coef = 1.0 # 0.5 for pen-human
gamma = 0.99
soft_target_tau = 5e-3

