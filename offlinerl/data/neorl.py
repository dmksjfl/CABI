import numpy as np
from loguru import logger

from offlinerl.utils.data import SampleBatch, get_scaler
from offlinerl.utils.data import BufferDataset, BufferDataloader

    
def load_neorl_buffer(data):
    buffer = SampleBatch(
        obs = data["obs"],
        obs_next = data["next_obs"],
        act = data["action"],
        rew = data["reward"],
        done = data["done"],
        )

    logger.info('obs shape: {}', buffer.obs.shape)
    logger.info('obs_next shape: {}', buffer.obs_next.shape)
    logger.info('act shape: {}', buffer.act.shape)
    logger.info('rew shape: {}', buffer.rew.shape)
    logger.info('done shape: {}', buffer.done.shape)
    logger.info('Episode reward: {}', buffer.rew.sum() /np.sum(buffer.done) )
    logger.info('Number of terminals on: {}', np.sum(buffer.done))
    
    return buffer
