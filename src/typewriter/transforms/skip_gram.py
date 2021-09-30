import numpy as np


class SkipGram(object):
    def __init__(self, window=1):
        assert window > 0
        self.window = window

    def output_length(self, seqlen):
        return self.window * (2 * seqlen - self.window - 1)

    def __call__(self, sequence: np.ndarray):
        seqlen = len(sequence)
        assert seqlen >= self.window + 1

        outlen = self.output_length(seqlen)

        text_indices = []
        context_indices = []

        for i_text in range(seqlen):
            for i_context in range(max(i_text - self.window, 0), i_text):
                text_indices.append(i_text)
                context_indices.append(i_context)
            for i_context in range(i_text + 1, min(i_text + 1 + self.window, seqlen)):
                text_indices.append(i_text)
                context_indices.append(i_context)

        text = sequence[text_indices].reshape(outlen, -1)
        context = sequence[context_indices].reshape(outlen, -1)

        return text, context
