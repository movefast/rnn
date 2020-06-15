import random
import numpy as np
from buffer.sum_tree import SumTree
# import replay_buffer_old
# from replay_buffer_old import Transition
from buffer.replay_buffer import Transition


class Memory:  # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6
    beta = 0.4  
    beta_increment_per_sampling = 0.001

    def __init__(self, capacity):
        self.tree = SumTree(capacity)
        self.capacity = capacity

    def _get_priority(self, error):
        return (np.abs(error) + self.e) ** self.a

    def add(self, error, *args):
        p = self._get_priority(error)
        if p == 0:
            import pdb; pdb.set_trace()
        self.tree.add(p, Transition(*args))

    def sample(self, n):
        batch = []
        idxs = []
        segment = self.tree.total() / n
        priorities = []

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            priorities.append(p)
            batch.append(data)
            idxs.append(idx)

        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxs, is_weight

    # most important state => last state
    # def sample_successive(self, n):
    #     assert self.tree.n_entries >= n, "we don't have long enough trajectory to sample from"

    #     self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

    #     while True:
    #         s = random.uniform(0, self.tree.total())
    #         idx = self.tree._retrieve(0, s) - n
    #         if idx >= 0:
    #             break
    #     dataIdx = idx - self.tree.capacity + 1
    #     idxes, priorities, batch = list(range(idx,idx+n)), self.tree.tree[idx:idx+n], self.tree.data[dataIdx:dataIdx+n]

    #     sampling_probabilities = priorities / self.tree.total()
    #     is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
    #     is_weight /= is_weight.max()

    #     return batch, idxes, is_weight

    def sample_successive(self, n):
        assert self.tree.n_entries >= n, "we don't have long enough trajectory to sample from"

        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])

        while True:
            s = random.uniform(0, self.tree.total())
            idx = self.tree._retrieve(0, s) - n + 1
            dataIdx = idx - self.tree.capacity + 1
            if dataIdx >= 0:
                break
        idxes, priorities, batch = list(range(idx,idx+n)), self.tree.tree[idx:idx+n], self.tree.data[dataIdx:dataIdx+n]
        # import pdb; pdb.set_trace()
        sampling_probabilities = priorities / self.tree.total()
        is_weight = np.power(self.tree.n_entries * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()

        return batch, idxes, is_weight

    def update(self, idx, error):
        p = self._get_priority(error)
        self.tree.update(idx, p)

    def __len__(self):
        return self.tree.n_entries
