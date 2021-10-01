import numpy as np
from typewriter.values.characters import Characters


class OneHot(object):
    def __init__(self, characters: Characters):
        assert characters
        self.characters = characters.copy()
        self.char2idx = {c: i for i, c in enumerate(characters.list())}

    def __call__(self, text):
        indices = np.array([self.char2idx[c] for c in text])
        return np.eye(len(self.characters), dtype=np.float32)[indices]
