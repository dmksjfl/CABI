# CABI: confidence-aware bidirectional model-based imagination

import torch
from torch.nn import Module
import torch.nn as nn
import numpy as np
from copy import deepcopy
from loguru import logger

from tianshou.data import Batch

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.net.common import MLP, Net
from offlinerl.utils.net.tanhpolicy import TanhGaussianPolicy
from offlinerl.utils.exp import setup_seed

from offlinerl.utils.data import ModelBuffer
from offlinerl.utils.net.model.ensemble import EnsembleTransition

from torch.autograd import Variable
from tensorboardX import SummaryWriter
import os

import torch.nn.functional as F

# Vanilla Variational Auto-Encoder 
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, latent_dim, max_action, device):
        super(VAE, self).__init__()
        self.e1 = nn.Linear(state_dim + action_dim, 750)
        self.e2 = nn.Linear(750, 750)

        self.mean = nn.Linear(750, latent_dim)
        self.log_std = nn.Linear(750, latent_dim)

        self.d1 = nn.Linear(state_dim + latent_dim, 750)
        self.d2 = nn.Linear(750, 750)
        self.d3 = nn.Linear(750, action_dim)

        self.max_action = max_action
        self.latent_dim = latent_dim
        self.device = device


    def forward(self, state, action):
        z = F.relu(self.e1(torch.cat([state, action], 1)))
        z = F.relu(self.e2(z))

        mean = self.mean(z)
        # Clamped for numerical stability 
        log_std = self.log_std(z).clamp(-4, 15)
        std = torch.exp(log_std)
        z = mean + std * torch.randn_like(std)
        
        u = self.decode(state, z)

        return u, mean, std


    def decode(self, state, z=None):
        # When sampling from the VAE, the latent vector is clipped to [-0.5, 0.5]
        if z is None:
            z = torch.randn((state.shape[0], self.latent_dim)).to(self.device).clamp(-0.5,0.5)

        a = F.relu(self.d1(torch.cat([state, z], 1)))
        a = F.relu(self.d2(a))
        return self.max_action * torch.tanh(self.d3(a))

def algo_init(args):
    logger.info('Run algo_init function')

    setup_seed(args['seed'])
    
    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
    elif "task" in args.keys():
        from offlinerl.utils.env import get_env_shape, get_env_action_range
        obs_shape, action_shape = get_env_shape(args['task'])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
        max_action, min_action = get_env_action_range(args['task'])
        args["max_action"] = max_action
    else:
        raise NotImplementedError
    
    forward_transition = EnsembleTransition(obs_shape, action_shape, args['hidden_layer_size'], args['transition_layers'], args['transition_init_num']).to(args['device'])
    forward_transition_optim = torch.optim.Adam(forward_transition.parameters(), lr=args['transition_lr'], weight_decay=0.000075)

    backward_transition = EnsembleTransition(obs_shape, action_shape, args['hidden_layer_size'], args['transition_layers'], args['transition_init_num']).to(args['device'])
    backward_transition_optim = torch.optim.Adam(backward_transition.parameters(), lr=args['transition_lr'], weight_decay=0.000075)

    ## forward variational auto-encoder
    latent_dim = action_shape * 2
    fvae = VAE(obs_shape, action_shape, latent_dim, max_action, args['device']).to(args['device'])
    fvae_optim = torch.optim.Adam(fvae.parameters())

    ## backward variational auto-encoder
    bvae = VAE(obs_shape, action_shape, latent_dim, max_action, args['device']).to(args['device'])
    bvae_optim = torch.optim.Adam(bvae.parameters())

    return {
        "forward_transition" : {"net" : forward_transition, "opt" : forward_transition_optim},
        "backward_transition" : {"net" : backward_transition, "opt" : backward_transition_optim},
        "fvae" : {"net" : fvae, "opt" : fvae_optim},
        "bvae" : {"net" : bvae, "opt" : bvae_optim},
    }

class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args

        self.forward_transition = algo_init['forward_transition']['net']
        self.forward_transition_optim = algo_init['forward_transition']['opt']
        self.forward_transition_optim_secheduler = torch.optim.lr_scheduler.ExponentialLR(self.forward_transition_optim, gamma=0.99)
        self.backward_transition = algo_init['backward_transition']['net']
        self.backward_transition_optim = algo_init['backward_transition']['opt']
        self.backward_transition_optim_secheduler = torch.optim.lr_scheduler.ExponentialLR(self.backward_transition_optim, gamma=0.99)
        self.selected_transitions = None

        self.fvae = algo_init['fvae']['net']
        self.fvae_optim = algo_init['fvae']['opt']
        self.bvae = algo_init['bvae']['net']
        self.bvae_optim = algo_init['bvae']['opt']

        self.device = args['device']
        self.forward_transition_train_epoch = args['forward_train_epoch']
        self.backward_transition_train_epoch = args['backward_train_epoch']
        
    def train(self, train_buffer, val_buffer):
        if self.args['dynamics_path'] is not None:
            self.forward_transition = torch.load(self.args['dynamics_path'], map_location='cpu').to(self.device)
            self.backward_transition = torch.load(self.args['dynamics_path'], map_location='cpu').to(self.device)
        else:
            self.train_transition(train_buffer)
        self.train_vae(train_buffer)  ## train VAE
        self.forward_transition.requires_grad_(False)
        self.backward_transition.requires_grad_(False)
        batch = self.train_bi_directional_model(train_buffer, val_buffer, 
                                   self.forward_transition, self.backward_transition, 
                                   self.fvae, self.bvae)
        return batch
    
    def get_policy(self):
        pass

    def train_transition(self, buffer):
        data_size = len(buffer.rew)
        val_size = min(int(data_size * 0.2) + 1, 1000)
        train_size = data_size - val_size
        train_splits, val_splits = torch.utils.data.random_split(range(data_size), (train_size, val_size))
        train_buffer = buffer[train_splits.indices]
        valdata = buffer[val_splits.indices]
        batch_size = self.args['transition_batch_size']

        forward_val_losses = [float('inf') for i in range(self.forward_transition.ensemble_size)]
        backward_val_losses = [float('inf') for i in range(self.backward_transition.ensemble_size)]

        epoch = 0
        forward_cnt = 0
        backward_cnt = 0
        with_cnt = self.args['with_cnt']
        ## train forward transition
        while True:
            epoch += 1
            idxs = np.random.randint(train_buffer.shape[0], size=[self.forward_transition.ensemble_size, train_buffer.shape[0]])
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                batch = train_buffer[batch_idxs]
                self._train_forward_transition(self.forward_transition, batch, self.forward_transition_optim)
            forward_new_val_losses = self._eval_forward_transition(self.forward_transition, valdata)
            print('Forward validation loss: ', forward_new_val_losses)

            forward_indexes = []
            for i, new_loss, old_loss in zip(range(len(forward_val_losses)), forward_new_val_losses, forward_val_losses):
                if new_loss < old_loss:
                    forward_indexes.append(i)
                    forward_val_losses[i] = new_loss

            if len(forward_indexes) > 0:
                self.forward_transition.update_save(forward_indexes)
                forward_cnt = 0
            else:
                forward_cnt += 1

            if with_cnt:
                if forward_cnt >= 5:
                    break
            else:
                if epoch >= self.forward_transition_train_epoch:
                    break

            self.forward_transition_optim_secheduler.step()
        
        while True:
            epoch += 1
            idxs = np.random.randint(train_buffer.shape[0], size=[self.backward_transition.ensemble_size, train_buffer.shape[0]])
            for batch_num in range(int(np.ceil(idxs.shape[-1] / batch_size))):
                batch_idxs = idxs[:, batch_num * batch_size:(batch_num + 1) * batch_size]
                batch = train_buffer[batch_idxs]
                self._train_backward_transition(self.backward_transition, batch, self.backward_transition_optim)
            backward_new_val_losses = self._eval_backward_transition(self.backward_transition, valdata)
            print('Backward validation loss:', backward_new_val_losses)

            backward_indexes = []
            
            for i, new_loss, old_loss in zip(range(len(backward_val_losses)), backward_new_val_losses, backward_val_losses):
                if new_loss < old_loss:
                    backward_indexes.append(i)
                    backward_val_losses[i] = new_loss

            if len(backward_indexes) > 0:
                self.backward_transition.update_save(backward_indexes)
                backward_cnt = 0
            else:
                backward_cnt += 1

            if with_cnt:
                if backward_cnt >= 5:
                    break
            else:
                if epoch >= self.backward_transition_train_epoch:
                    break

            self.backward_transition_optim_secheduler.step()
        
        forward_indexes = self._select_best_indexes(forward_val_losses, n=self.args['transition_select_num'])
        self.forward_transition.set_select(forward_indexes)
        backward_indexes = self._select_best_indexes(backward_val_losses, n=self.args['transition_select_num'])
        self.backward_transition.set_select(backward_indexes)
        return self.forward_transition, self.backward_transition
    
    def train_vae(self, buffer):
        for i in range(self.args['vae_train_epoch']):
            batch = buffer.sample(self.args['transition_batch_size'])
            self._train_forward_vae(batch, self.fvae, self.fvae_optim)
            self._train_backward_vae(batch, self.bvae, self.bvae_optim)

    def train_bi_directional_model(self, train_buffer, val_buffer, forward_transition, backward_transition, fvae, bvae):
        real_batch_size = int(train_buffer.obs.shape[0] * self.args['real_data_ratio'])
        model_batch_size =  int(self.args['buffer_size'] * (1 - self.args['real_data_ratio']))
        
        model_buffer = ModelBuffer(self.args['buffer_size'])

        for epoch in range(self.args['model_epoch']):
            # collect data
            with torch.no_grad():
                ## bi-directional multi-step prediction with double check
                ## forward imagination
                obs = train_buffer.sample(int(self.args['data_collection_per_epoch']))['obs']
                obs = torch.tensor(obs, device=self.device)
                for t in range(self.args['horizon']):
                    # sample from variational auto-encoder
                    action = self.fvae.decode(obs)
                    obs_action = torch.cat([obs, action], dim=-1)
                    next_obs_dists = forward_transition(obs_action)
                    next_obses = next_obs_dists.sample()
                    rewards = next_obses[:, :, -1:]
                    next_obses = next_obses[:, :, :-1]

                    next_obses_mode = next_obs_dists.mean[:, :, :-1]
                    next_obs_mean = torch.mean(next_obses_mode, dim=0)
                    diff = next_obses_mode - next_obs_mean
                    disagreement_uncertainty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0]
                    aleatoric_uncertainty = torch.max(torch.norm(next_obs_dists.stddev, dim=-1, keepdim=True), dim=0)[0]
                    uncertainty = disagreement_uncertainty if self.args['uncertainty_mode'] == 'disagreement' else aleatoric_uncertainty

                    model_indexes = np.random.randint(0, next_obses.shape[0], size=(obs.shape[0]))
                    next_obs = next_obses[model_indexes, np.arange(obs.shape[0])]

                    reward = rewards[model_indexes, np.arange(obs.shape[0])]

                    reward -= self.args['rew_penalty'] * uncertainty
                    
                    print('forward average reward:', reward.mean().item())

                    dones = torch.zeros_like(reward)

                    # backward state
                    next_obs_action = torch.cat([next_obs, action], dim=-1)
                    previous_obs_dists = backward_transition(next_obs_action)
                    previous_obses = previous_obs_dists.sample()
                    previous_obses = previous_obses[:, :, :-1]
                    previous_obs = previous_obses[model_indexes, np.arange(next_obs.shape[0])]
                    forward_diff = -torch.mean((obs - previous_obs)**2,dim=-1)
                    bfdiff, forward_top_index = torch.topk(forward_diff, k = int(self.args['top_k']*int(self.args['data_collection_per_epoch'])))

                    ## select high confidence states
                    conf_obs = torch.index_select(obs, dim=0, index=forward_top_index)
                    conf_action = torch.index_select(action, dim=0, index=forward_top_index)
                    conf_rew = torch.index_select(reward, dim=0, index=forward_top_index)
                    conf_dones = torch.index_select(dones, dim=0, index=forward_top_index)
                    conf_next_obs = torch.index_select(next_obs, dim=0, index=forward_top_index)

                    batch_data = Batch({
                        "obs" : conf_obs.cpu(),
                        "act" : conf_action.cpu(),
                        "rew" : conf_rew.cpu(),
                        "done" : conf_dones.cpu(),
                        "obs_next" : conf_next_obs.cpu(),
                    })
                    model_buffer.put(batch_data)

                    obs = next_obs
                
                ## backward imagination
                next_obs = train_buffer.sample(int(self.args['data_collection_per_epoch']))['obs_next']
                next_obs = torch.tensor(next_obs, device=self.device)
                backward_diff = []
                for t in range(self.args['horizon']):
                    # sample from variational auto-encoder
                    action = self.bvae.decode(next_obs)
                    next_obs_action = torch.cat([next_obs, action], dim=-1)
                    pre_obs_dists = backward_transition(next_obs_action)
                    pre_obses = pre_obs_dists.sample()
                    rewards = pre_obses[:, :, -1:]
                    pre_obses = pre_obses[:, :, :-1]

                    pre_obses_mode = pre_obs_dists.mean[:, :, :-1]
                    pre_obs_mean = torch.mean(pre_obses_mode, dim=0)
                    diff = pre_obses_mode - pre_obs_mean
                    disagreement_uncertainty = torch.max(torch.norm(diff, dim=-1, keepdim=True), dim=0)[0]
                    aleatoric_uncertainty = torch.max(torch.norm(pre_obs_dists.stddev, dim=-1, keepdim=True), dim=0)[0]
                    uncertainty = disagreement_uncertainty if self.args['uncertainty_mode'] == 'disagreement' else aleatoric_uncertainty

                    back_model_indexes = np.random.randint(0, pre_obses.shape[0], size=(next_obs.shape[0]))
                    pre_obs = pre_obses[back_model_indexes, np.arange(next_obs.shape[0])]
                    reward = rewards[back_model_indexes, np.arange(next_obs.shape[0])]
                    reward -= self.args['rew_penalty'] * uncertainty
                    
                    print('backward average reward:', reward.mean().item())

                    dones = torch.zeros_like(reward)

                    # forward state
                    pre_obs_action = torch.cat([pre_obs, action], dim=-1)
                    forward_obs_dists = forward_transition(pre_obs_action)
                    forward_obses = forward_obs_dists.sample()
                    forward_obses = forward_obses[:, :, :-1]
                    forward_obs = forward_obses[back_model_indexes, np.arange(pre_obs.shape[0])]
                    backward_diff = -torch.mean((forward_obs - next_obs)**2,dim=-1)
                    bfdiff, backward_top_index = torch.topk(backward_diff, k = int(self.args['top_k']*int(self.args['data_collection_per_epoch'])))

                    ## select high confidence states
                    conf_obs = torch.index_select(pre_obs, dim=0, index=backward_top_index)
                    conf_action = torch.index_select(action, dim=0, index=backward_top_index)
                    conf_rew = torch.index_select(reward, dim=0, index=backward_top_index)
                    conf_dones = torch.index_select(dones, dim=0, index=backward_top_index)
                    conf_next_obs = torch.index_select(next_obs, dim=0, index=backward_top_index)

                    batch_data = Batch({
                        "obs" : conf_obs.cpu(),
                        "act" : conf_action.cpu(),
                        "rew" : conf_rew.cpu(),
                        "done" : conf_dones.cpu(),
                        "obs_next" : conf_next_obs.cpu(),
                    })
                    model_buffer.put(batch_data)

                    next_obs = pre_obs

        return model_buffer

    def _select_best_indexes(self, metrics, n):
        pairs = [(metric, index) for metric, index in zip(metrics, range(len(metrics)))]
        pairs = sorted(pairs, key=lambda x: x[0])
        selected_indexes = [pairs[i][1] for i in range(n)]
        return selected_indexes

    def _train_forward_transition(self, transition, data, optim):
        data.to_torch(device=self.device)
        dist = transition(torch.cat([data['obs'], data['act']], dim=-1))
        loss = - dist.log_prob(torch.cat([data['obs_next'], data['rew']], dim=-1))
        loss = loss.mean()

        loss = loss + 0.01 * transition.max_logstd.mean() - 0.01 * transition.min_logstd.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        
    def _eval_forward_transition(self, transition, valdata):
        with torch.no_grad():
            valdata.to_torch(device=self.device)
            dist = transition(torch.cat([valdata['obs'], valdata['act']], dim=-1))
            loss = ((dist.mean - torch.cat([valdata['obs_next'], valdata['rew']], dim=-1)) ** 2).mean(dim=(1,2))
            return list(loss.cpu().numpy())
    
    def _train_backward_transition(self, transition, data, optim):
        data.to_torch(device=self.device)
        dist = transition(torch.cat([data['obs_next'], data['act']], dim=-1))
        loss = - dist.log_prob(torch.cat([data['obs'], data['rew']], dim=-1))
        loss = loss.mean()

        loss = loss + 0.01 * transition.max_logstd.mean() - 0.01 * transition.min_logstd.mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        
    def _eval_backward_transition(self, transition, valdata):
        with torch.no_grad():
            valdata.to_torch(device=self.device)
            dist = transition(torch.cat([valdata['obs_next'], valdata['act']], dim=-1))
            loss = ((dist.mean - torch.cat([valdata['obs'], valdata['rew']], dim=-1)) ** 2).mean(dim=(1,2))
            return list(loss.cpu().numpy())
    
    def _train_forward_vae(self, data, fvae, fvae_optim):
        data.to_torch(device=self.device)
        state = data['obs']
        action = data['act']
        # Variational Auto-Encoder Training
        recon, mean, std = fvae(state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        fvae_loss = recon_loss + 0.5 * KL_loss

        fvae_optim.zero_grad()
        fvae_loss.backward()
        fvae_optim.step()
    
    def _train_backward_vae(self, data, bvae, bvae_optim):
        data.to_torch(device=self.device)
        next_state = data['obs_next']
        action = data['act']
        # Variational Auto-Encoder Training
        recon, mean, std = bvae(next_state, action)
        recon_loss = F.mse_loss(recon, action)
        KL_loss	= -0.5 * (1 + torch.log(std.pow(2)) - mean.pow(2) - std.pow(2)).mean()
        bvae_loss = recon_loss + 0.5 * KL_loss

        bvae_optim.zero_grad()
        bvae_loss.backward()
        bvae_optim.step()