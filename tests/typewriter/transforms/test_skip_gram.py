import numpy as np

from typewriter.transforms.skip_gram import SkipGram


class TestSkipGram:
    @staticmethod
    def test_skip_gram():
        original = np.array(
            [
                [10, 20],
                [20, 30],
                [30, 40],
                [40, 50],
                [50, 60],
            ],
            dtype=np.float32,
        )
        context_expected = np.array(
            [
                [10 + 30, 20 + 40],
                [20 + 40, 30 + 50],
                [30 + 50, 40 + 60],
            ],
            dtype=np.float32,
        )
        text_expected = np.array(
            [
                [20, 30],
                [30, 40],
                [40, 50],
            ],
            dtype=np.float32,
        )

        transform = SkipGram(left=1, skip=1, right=1)
        output_context, output_text = transform(original)

        assert (output_context == context_expected).all()
        assert (output_text == text_expected).all()
