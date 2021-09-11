import numpy as np

from typewriter.transforms.skip_gram import SkipGram


class TestSkipGram:
    @staticmethod
    def test_skip_gram():
        original = np.array([
            [10, 20],
            [20, 30],
            [30, 40],
            [40, 50],
            [50, 60],
        ])

        expected = np.array([
            [
                [10, 20],
                [30, 40],
            ],
            [
                [20, 30],
                [40, 50],
            ],
            [
                [30, 40],
                [50, 60],
            ],
        ])

        transform = SkipGram(left=1, skip=1, right=1)

        output = transform(original)

        assert (output == expected).all()
