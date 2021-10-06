import numpy as np
from functools import lru_cache


class SkipGram(object):
    def __init__(self, window=1):
        assert window > 0
        self.window = window

    def output_length(self, seqlen):
        return self.window * (2 * seqlen - self.window - 1)

    @staticmethod
    @lru_cache(maxsize=1024)
    def build_indices(seqlen: int, window: int):

        text_indices = []
        context_indices = []

        for i_text in range(seqlen):
            for i_context in range(max(i_text - window, 0), i_text):
                text_indices.append(i_text)
                context_indices.append(i_context)
            for i_context in range(i_text + 1, min(i_text + 1 + window, seqlen)):
                text_indices.append(i_text)
                context_indices.append(i_context)
        return np.array(text_indices), np.array(context_indices)

    def __call__(self, sequence: np.ndarray):
        seqlen = len(sequence)
        assert seqlen >= self.window + 1

        text_indices, context_indices = self.build_indices(seqlen, self.window)

        text = sequence[text_indices]
        context = sequence[context_indices]

        return text, context
