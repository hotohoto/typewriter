import numpy as np


class SkipGram(object):
    def __init__(self, left=1, skip=1, right=1):
        assert left > 0
        assert skip > 0
        assert right > 0
        self.left = left
        self.skip = skip
        self.right = right

    @property
    def window_size(self):
        return self.left + self.skip + self.right

    @property
    def size(self):
        return self.left + self.right

    def length(self, seqlen):
        return seqlen - self.window_size + 1

    def __call__(self, sequence: np.ndarray):
        assert len(sequence) >= self.window_size
        start = np.array(list(range(self.left)) + list(range(self.left + self.skip, self.window_size)))

        length = self.length(len(sequence))

        indices = np.empty((length, self.size), dtype=int)

        for i in range(length):
            indices[i] = start + i

        return sequence[indices.flatten()].reshape(length, self.size, -1)
