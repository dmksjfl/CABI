import fire

from offlinerl.algo import algo_select
from offlinerl.data.d4rl import load_d4rl_buffer
from offlinerl.evaluation import OnlineCallBackFunction

import torch
import numpy as np
import random


def run_algo(**kwargs):
    algo_init_fn, algo_trainer_obj, algo_config = algo_select(kwargs)
    train_buffer, weights = load_d4rl_buffer(algo_config["task"], algo_config['use_per'])

    ## setup seed
    torch.manual_seed(algo_config["seed"])
    np.random.seed(algo_config["seed"])
    torch.cuda.manual_seed_all(algo_config["seed"])
    random.seed(algo_config["seed"])

    if algo_config['use_cabi']:
        from offlinerl.algo.modelbase import cabi
        from offlinerl.config.algo import cabi_config
        from offlinerl.utils.config import parse_config
        cabi_config = parse_config(cabi_config)
        cabi_config.update(kwargs)
        data_algo_init = cabi.algo_init(cabi_config)
        data_algo_trainer = cabi.AlgoTrainer(data_algo_init, cabi_config)
        model_buffer = data_algo_trainer.train(train_buffer=train_buffer, val_buffer=None)
    algo_init = algo_init_fn(algo_config)
    algo_trainer = algo_trainer_obj(algo_init, algo_config)
    callback = OnlineCallBackFunction()
    if algo_config['use_cabi']:
        callback.initialize(train_buffer=train_buffer, val_buffer=None, task=algo_config["task"])
        algo_trainer.train(train_buffer=train_buffer, val_buffer=model_buffer, callback_fn=callback, weights=None)
    else:
        callback.initialize(train_buffer=train_buffer, val_buffer=None, task=algo_config["task"])
        algo_trainer.train(train_buffer, None, callback_fn=callback)

if __name__ == "__main__":
    fire.Fire(run_algo)
    