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
        context_start_index = np.array(
            list(range(self.left)) + list(range(self.left + self.skip, self.window_size)),
            dtype=np.float32,
        )
        text_start_index = np.array(list(range(self.left, self.left + self.skip)), dtype=np.float32)

        length = self.length(len(sequence))

        context_indices = np.empty((length, self.size), dtype=int)
        text_indices = np.empty((length, self.skip), dtype=int)

        for i in range(length):
            context_indices[i] = context_start_index + i
            text_indices[i] = text_start_index + i

        context = sequence[context_indices.flatten()].reshape(length, self.size, -1).sum(axis=1)
        text = sequence[text_indices.flatten()].reshape(length, self.skip, -1).sum(axis=1)

        return context, text
