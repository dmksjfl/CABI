import torch
import numpy as np
import random
import math

from sklearn.preprocessing import MinMaxScaler
from tianshou.data import Batch
from tianshou.data import to_torch, to_torch_as, to_numpy
from torch.utils.data import dataset
from torch.utils.data import dataloader


def to_array_as(x, y):    
    if isinstance(x, torch.Tensor) and isinstance(y, np.ndarray):
        return to_numpy(x)
    elif isinstance(x, np.ndarray) and isinstance(y, torch.Tensor):
        return to_torch_as(x)
    else:
        return x
    
class BufferDataset(dataset.Dataset):
    def __init__(self, buffer, batch_size=256):
        self.buffer = buffer
        self.batch_size = batch_size
        self.length = len(self.buffer)
        
    def __getitem__(self, index):
        indices = np.random.randint(0, self.length, self.batch_size)
        data = self.buffer[indices]
        
        return data
        
    def __len__(self):
        return self.length
    
    
class BufferDataloader(dataloader.DataLoader):        
    def sample(self, batch_size=None): 
        if not hasattr(self, 'buffer_loader') or batch_size != self.buffer_loader._dataset.batch_size:
            if not hasattr(self, 'buffer_loader'):
                self.buffer_loader = self.__iter__()
            elif batch_size is None:
                pass
            else:
                self.dataset.batch_size = batch_size
                self.buffer_loader = self.__iter__()
        try:
            return self.buffer_loader.__next__()
        except:
            self.buffer_loader = self.__iter__()
            return self.buffer_loader.__next__()
    
class SampleBatch(Batch):

    def sample(self, batch_size, proportional=None):
        length = len(self)
        assert 1 <= batch_size
        
        if proportional is not None:
            indices = proportional.select(batch_size)
        else:
            indices = np.random.randint(0, length, batch_size)
        
        return self[indices]

def sample(batch : Batch, batch_size : int):
    length = len(batch)
    assert 1 <= batch_size
    
    indices = np.random.randint(0, length, batch_size)

    return batch[indices]


def get_scaler(data):
    scaler = MinMaxScaler((-1,1))
    scaler.fit(data)
    
    return scaler

class ModelBuffer:
    def __init__(self, buffer_size):
        self.data = None
        self.buffer_size = int(buffer_size)

    def put(self, batch_data):
        batch_data.to_torch(device='cpu')

        if self.data is None:
            self.data = batch_data
        else:
            self.data.cat_(batch_data)
        
        if len(self) > self.buffer_size:
            self.data = self.data[len(self) - self.buffer_size : ]

    def __len__(self):
        if self.data is None: return 0
        return self.data.shape[0]

    def sample(self, batch_size):
        indexes = np.random.randint(0, len(self), size=(batch_size))
        return self.data[indexes]


class SumTree(object):
	def __init__(self, max_size):
		self.max_size = max_size
		self.tree_level = math.ceil(math.log(max_size + 1, 2)) + 1
		self.tree_size = 2 ** self.tree_level - 1
		self.tree = [0. for _ in range(self.tree_size)]
		self.size = 0
		self.cursor = 0

	def add(self, value):
		index = self.cursor
		self.cursor = (self.cursor + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)
		self.val_update(index, value)

	def get_val(self, index):
		tree_index = 2 ** (self.tree_level - 1) - 1 + index
		return self.tree[tree_index]

	def val_update(self, index, value):
		tree_index = 2 ** (self.tree_level - 1) - 1 + index
		diff = value - self.tree[tree_index]
		self.reconstruct(tree_index, diff)

	def reconstruct(self, tindex, diff):
		self.tree[tindex] += diff
		if not tindex == 0:
			tindex = int((tindex - 1) / 2)
			self.reconstruct(tindex, diff)

	def find(self, value, norm=True):
		pre_value = value
		if norm:
			value *= self.tree[0]
		list = []
		return self._find(value, 0, pre_value, list)

	def _find(self, value, index, r, list):
		if 2 ** (self.tree_level - 1) - 1 <= index:
			if index - (2 ** (self.tree_level - 1) - 1) >= self.size:
				print('!!!!!')
				print(index, value, self.tree[0], r)
				print(list)
				index = (2 ** (self.tree_level - 1) - 1) + random.randint(0, self.size)
				#index = (2 ** (self.tree_level - 1) - 1)
			return self.tree[index], index - (2 ** (self.tree_level - 1) - 1)

		left = self.tree[2 * index + 1]
		list.append(left)

		if value <= left + 1e-8:
			return self._find(value, 2 * index + 1, r, list)
		else:
			return self._find(value - left, 2 * (index + 1), r, list)

	def print_tree(self):
		for k in range(1, self.tree_level + 1):
			for j in range(2 ** (k - 1) - 1, 2 ** k - 1):
				print(self.tree[j])
			print()

	def filled_size(self):
		return self.size

class Experience(object):
	""" The class represents prioritized experience replay buffer.
	The class has functions: store samples, pick samples with
	probability in proportion to sample's priority, update
	each sample's priority, reset alpha.
	see https://arxiv.org/pdf/1511.05952.pdf .
	"""

	def __init__(self, memory_size, alpha=1):
		self.tree = SumTree(memory_size)
		self.memory_size = memory_size
		self.alpha = alpha

	def add(self, priority):
		self.tree.add(priority ** self.alpha)

	def select(self, batch_size):

		if self.tree.filled_size() < batch_size:
			return None

		indices = []
		priorities = []
		for _ in range(batch_size):
			r = random.random()
			priority, index = self.tree.find(r)
			priorities.append(priority ** (1. / self.alpha))
			indices.append(index)
			self.priority_update([index], [0])  # To avoid duplicating

		self.priority_update(indices, priorities)  # Revert priorities

		return indices

	def priority_update(self, indices, priorities):
		""" The methods update samples's priority.
		Parameters
		----------
		indices :
			list of sample indices
		"""
		for i, p in zip(indices, priorities):
			self.tree.val_update(i, p ** self.alpha)