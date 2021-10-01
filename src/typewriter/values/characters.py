import typing as t

import numpy as np
import numpy.testing as npt


class Characters:
    def __init__(self, probs: t.Dict[str, int]):

        values = np.array(list(probs.values()))
        npt.assert_almost_equal(values.sum(), 1)
        assert (0 <= values).all() and (values <= 1).all()

        self.probs = probs.copy()
        self.keys = sorted(probs.keys())

    def list(self):
        return self.keys.copy()

    def to_dict(self):
        return self.probs

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Characters) and self.to_dict() == o.to_dict()

    def __len__(self):
        return len(self.keys)

    def copy(self):
        return Characters.from_dict(self.to_dict())

    @staticmethod
    def from_dict(data):
        return Characters(probs=data)
