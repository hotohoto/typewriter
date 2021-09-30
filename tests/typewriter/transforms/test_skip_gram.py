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
        text_expected = np.array(
            [
                [10, 20],
                [20, 30],
                [20, 30],
                [30, 40],
                [30, 40],
                [40, 50],
                [40, 50],
                [50, 60],
            ],
            dtype=np.float32,
        )
        context_expected = np.array(
            [
                [20, 30],
                [10, 20],
                [30, 40],
                [20, 30],
                [40, 50],
                [30, 40],
                [50, 60],
                [40, 50],
            ],
            dtype=np.float32,
        )

        transform = SkipGram(window=1)
        assert transform.output_length(3) == 4
        assert transform.output_length(5) == 8

        output_text, output_context = transform(original)

        assert (output_context == context_expected).all()
        assert (output_text == text_expected).all()
