# TD3_BC
import copy

import torch
import numpy as np
from torch import nn
from torch import optim
from loguru import logger

from offlinerl.algo.base import BaseAlgo
from offlinerl.utils.data import to_torch
from offlinerl.utils.net.common import Net
from offlinerl.utils.net.continuous import Critic
from offlinerl.utils.net.continuous import Actor
from offlinerl.utils.exp import setup_seed

from tensorboardX import SummaryWriter
import os
from offlinerl.evaluation.d4rl import d4rl_score

def algo_init(args):
    logger.info('Run algo_init function')

    setup_seed(args['seed'])
    
    if args["obs_shape"] and args["action_shape"]:
        obs_shape, action_shape = args["obs_shape"], args["action_shape"]
    elif "task" in args.keys():
        from offlinerl.utils.env import get_env_shape
        obs_shape, action_shape = get_env_shape(args['task'])
        args["obs_shape"], args["action_shape"] = obs_shape, action_shape
    else:
        raise NotImplementedError
    
    net_a = Net(layer_num = args['layer_num'], 
                     state_shape = obs_shape, 
                     hidden_layer_size = args['hidden_layer_size'])

    from offlinerl.utils.env import get_env_action_range
    action_max, action_min = get_env_action_range(args['task'])
    
    actor = Actor(preprocess_net = net_a,
                    action_shape = action_shape,
                    hidden_layer_size = args['hidden_layer_size'],
                    max_action = action_max,
                ).to(args['device'])
    
    actor_optim = optim.Adam(actor.parameters(), lr=args['actor_lr'])
    
    net_c1 = Net(layer_num = args['layer_num'],
                  state_shape = obs_shape,  
                  action_shape = action_shape,
                  concat = True, 
                  hidden_layer_size = args['hidden_layer_size'])
    critic1 = Critic(preprocess_net = net_c1,  
                     hidden_layer_size = args['hidden_layer_size'],
                    ).to(args['device'])
    critic1_optim = optim.Adam(critic1.parameters(), lr=args['critic_lr'])
    
    net_c2 = Net(layer_num = args['layer_num'],
                  state_shape = obs_shape,  
                  action_shape = action_shape,
                  concat = True, 
                  hidden_layer_size = args['hidden_layer_size'])
    critic2 = Critic(preprocess_net = net_c2, 
                     hidden_layer_size = args['hidden_layer_size'],
                    ).to(args['device'])
    critic2_optim = optim.Adam(critic2.parameters(), lr=args['critic_lr'])
        
    nets =  {
        "actor" : {"net" : actor, "opt" : actor_optim},
        "critic1" : {"net" : critic1, "opt" : critic1_optim},
        "critic2" : {"net" : critic2, "opt" : critic2_optim},
    }
        
    return nets


class AlgoTrainer(BaseAlgo):
    def __init__(self, algo_init, args):
        super(AlgoTrainer, self).__init__(args)
        self.args = args
        
        self.actor = algo_init["actor"]["net"]
        self.actor_opt = algo_init["actor"]["opt"]
        
        self.critic1 = algo_init["critic1"]["net"]
        self.critic1_opt = algo_init["critic1"]["opt"]
        self.critic2 = algo_init["critic2"]["net"]
        self.critic2_opt = algo_init["critic2"]["opt"]
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)
        self.actor_target = copy.deepcopy(self.actor)
            
        self.critic_criterion = nn.MSELoss()

        self.exploration_noise = args['exploration_noise']
        self.alpha = args['alpha']
        self.policy_noise = args['policy_noise']
        self.noise_clip = args['noise_clip']
        
        self._n_train_steps_total = 0
        self.device = args["device"]

        if not os.path.exists("{}".format(args['log_dir'])):
            os.makedirs("{}".format(args['log_dir']))
        self.writer = SummaryWriter('{}'.format(args['log_dir']))
        self.task = args['task'][5:]
        from offlinerl.utils.env import get_env_action_range
        action_max, action_min = get_env_action_range(args['task'])
        self.act_max = action_max
        self.act_min = action_min
        self.eps = 1e-3
        
    def _train(self, batch, mean, std):
        batch = to_torch(batch, torch.float, device=self.args["device"])
        rewards = batch.rew
        terminals = batch.done
        obs = batch.obs
        actions = batch.act
        next_obs = batch.obs_next

        ## normalization
        obs = (obs - mean)/(0.001 + std)
        next_obs = (next_obs - mean)/(0.001 + std)

        """
        QF Loss
        """

        with torch.no_grad():
            # Select action according to policy and add clipped noise
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_action = (self.actor_target(next_obs) + noise).clamp(-self.act_max, self.act_max)
            
            target_q = torch.min(
                self.critic1_target(next_obs, next_action),
                self.critic2_target(next_obs, next_action)
            )
            q_target = rewards + (1. - terminals) * self.args["discount"] * target_q

        q1_pred = self.critic1(obs, actions)
        q2_pred = self.critic2(obs, actions)
            
        critic1_loss = self.critic_criterion(q1_pred, q_target)
        critic2_loss = self.critic_criterion(q2_pred, q_target)

        """
        Update critic networks
        """
        self.critic1_opt.zero_grad()
        critic1_loss.backward(retain_graph=True)
        self.critic1_opt.step()

        self.critic2_opt.zero_grad()
        critic2_loss.backward()
        self.critic2_opt.step()

        if self._n_train_steps_total % 2 == 0:
            """
            Policy Loss
            """

            new_obs_actions = self.actor(obs)
            q_new_actions = self.critic1(obs, new_obs_actions)
            lmbda = self.alpha/q_new_actions.abs().mean().detach()
            
            policy_loss = -lmbda * q_new_actions.mean() + self.critic_criterion(new_obs_actions, actions)
            self.actor_opt.zero_grad()
            policy_loss.backward()
            self.actor_opt.step()

            """
            Soft Updates target network
            """
            self._sync_weight(self.critic1_target, self.critic1, self.args["soft_target_tau"])
            self._sync_weight(self.critic2_target, self.critic2, self.args["soft_target_tau"])
            self._sync_weight(self.actor_target, self.actor, self.args["soft_target_tau"])
    
        self._n_train_steps_total += 1

        
    def get_model(self):
        return self.actor
        
    def get_policy(self):
        return self.actor
    
    def train(self, train_buffer, val_buffer, callback_fn, weights=None):
        sur_obs = torch.from_numpy(train_buffer.obs).to(self.device)
        mean = torch.mean(sur_obs, dim=0, keepdim=True)
        std = torch.std(sur_obs, dim=0, keepdim=True)
        for epoch in range(1,self.args["max_epoch"]+1):
            for step in range(1,self.args["steps_per_epoch"]+1):
                real_batch_size = int(self.args["batch_size"] * self.args['real_data_ratio'])
                model_batch_size = int(self.args['batch_size'] - real_batch_size)
                train_data = train_buffer.sample(real_batch_size)
                model_data = val_buffer.sample(model_batch_size)
                train_data.cat_(model_data)
                self._train(train_data, mean, std)
            
            res = callback_fn(self.get_policy(), epoch, mean.cpu().numpy(), std.cpu().numpy(), normalize=self.args['normalize'])

            self.log_res(epoch, res)

            ## record statistics
            self.writer.add_scalar('test return', res['Reward_Mean_Env'], global_step = epoch)
            self.writer.add_scalar('score', d4rl_score(self.task, res['Reward_Mean_Env']), global_step = epoch)
            
        return self.get_policy()