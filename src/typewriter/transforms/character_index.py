import numpy as np
from typewriter.values.characters import Characters


class CharacterIndex(object):
    def __init__(self, characters: Characters):
        assert characters
        self.characters = characters.copy()
        self.char2idx = {c: i for i, c in enumerate(characters.list())}

    def __call__(self, text):
        return np.array([self.char2idx[c] for c in text])
