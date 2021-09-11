import typing as t

import numpy as np


class OneHot(object):
    def __init__(self, characters: t.List):
        assert characters
        self.characters = characters.copy()
        self.char2idx = {c: i for i, c in enumerate(characters)}

    def __call__(self, text):
        indices = np.array([self.char2idx[c] for c in text])
        return np.eye(len(self.characters))[indices]
