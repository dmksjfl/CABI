import os
import pickle
import copy as cp
import collections

import gym
import d4rl
import numpy as np
from loguru import logger

from offlinerl.utils.data import SampleBatch, Experience

def load_d4rl_buffer(task, use_per=False):
    env = gym.make(task[5:])
    if use_per:
        dataset = get_dataset_with_per_weight(env)
        weights = dataset['weights']
    else:
        dataset = d4rl.qlearning_dataset(env)
        weights = None

    buffer = SampleBatch(
        obs=dataset['observations'],
        obs_next=dataset['next_observations'],
        act=dataset['actions'],
        rew=np.expand_dims(np.squeeze(dataset['rewards']), 1),
        done=np.expand_dims(np.squeeze(dataset['terminals']), 1),
    )

    logger.info('obs shape: {}', buffer.obs.shape)
    logger.info('obs_next shape: {}', buffer.obs_next.shape)
    logger.info('act shape: {}', buffer.act.shape)
    logger.info('rew shape: {}', buffer.rew.shape)
    logger.info('done shape: {}', buffer.done.shape)
    logger.info('Episode reward: {}', buffer.rew.sum() /np.sum(buffer.done) )
    logger.info('Number of terminals on: {}', np.sum(buffer.done))
    return buffer, weights

# prioritized experience replay (PER)
def get_dataset_with_per_weight(env, is_render_traj=False):
    dataset = env.get_dataset()

    traj_list = processed_sequence_dataset(dataset, timeout_frame=False, done_frame=True)

    traj_end = []
    data = collections.defaultdict(list)

    for traj in traj_list:
        num_tuple = traj['rewards'].shape[0]
        traj['weights'] = cp.deepcopy(traj['rewards'])
        sum_return = np.sum(traj['rewards'])
        traj['weights'][:] = sum_return

        if is_render_traj:
            if sum_return > 0. and num_tuple > 100:
                for item in traj.keys():
                    data[item].append(traj[item][::-1])
                break
        else:
            for item in traj.keys():
                data[item].append(traj[item])

    for item in data.keys():
        data[item] = np.concatenate(data[item], axis=0)

    data['weights'] -= np.min(data['weights'])
    max_return = max(1., np.max(data['weights']))
    data['weights'] /= max_return
    data['weights'] = (0.001 + data['weights']) / (0.001 + 1.0)

    return data

def processed_sequence_dataset(dataset, timeout_frame=False, done_frame=False):

    N = dataset['rewards'].shape[0]
    obs_ = []
    next_obs_ = []
    action_ = []
    reward_ = []
    done_ = []
    traj_list = []

    # The newer version of the dataset adds an explicit
    # timeouts field. Keep old method for backwards compatability.
    use_timeouts = False
    if 'timeouts' in dataset:
        use_timeouts = True

    episode_step = 0
    for i in range(N - 1):
        obs = dataset['observations'][i].astype(np.float32)
        new_obs = dataset['observations'][i + 1].astype(np.float32)
        action = dataset['actions'][i].astype(np.float32)
        reward = dataset['rewards'][i].astype(np.float32)
        done_bool = bool(dataset['terminals'][i])

        if use_timeouts:
            final_timestep = dataset['timeouts'][i]
        else:
            final_timestep = (episode_step == env._max_episode_steps - 1)
        if (not timeout_frame) and final_timestep:
            # Skip this transition and don't apply terminals on the last step of an episode
            if episode_step > 0:
                traj = {
                    'observations': np.array(obs_),
                    'actions': np.array(action_),
                    'next_observations': np.array(next_obs_),
                    'rewards': np.array(reward_),
                    'terminals': np.array(done_),
                }
                traj_list.append(traj)
                obs_ = []
                next_obs_ = []
                action_ = []
                reward_ = []
                done_ = []

            episode_step = 0
            continue
        if (not done_frame) and done_bool:
            # Skip this transition and don't apply terminals on the last step of an episode
            if episode_step > 0:
                traj = {
                    'observations': np.array(obs_),
                    'actions': np.array(action_),
                    'next_observations': np.array(next_obs_),
                    'rewards': np.array(reward_),
                    'terminals': np.array(done_),
                }
                traj_list.append(traj)
                obs_ = []
                next_obs_ = []
                action_ = []
                reward_ = []
                done_ = []

            episode_step = 0
            continue
        if done_bool or final_timestep:
            obs_.append(obs)
            next_obs_.append(new_obs)
            action_.append(action)
            reward_.append(reward)
            done_.append(done_bool)
            episode_step += 1
            if episode_step > 0:
                traj = {
                    'observations': np.array(obs_),
                    'actions': np.array(action_),
                    'next_observations': np.array(next_obs_),
                    'rewards': np.array(reward_),
                    'terminals': np.array(done_),
                }
                traj_list.append(traj)
                obs_ = []
                next_obs_ = []
                action_ = []
                reward_ = []
                done_ = []

            episode_step = 0
            continue

        obs_.append(obs)
        next_obs_.append(new_obs)
        action_.append(action)
        reward_.append(reward)
        done_.append(done_bool)
        episode_step += 1

    if episode_step > 0:
        traj = {
            'observations': np.array(obs_),
            'actions': np.array(action_),
            'next_observations': np.array(next_obs_),
            'rewards': np.array(reward_),
            'terminals': np.array(done_),
        }
        traj_list.append(traj)

    return traj_list