import numpy as np
import pytest
import torch
from typewriter.usecases.char2vec import Char2Vec, char2vec


class TestChar2Vec:
    @staticmethod
    def test_char2vec():

        batch_size = 4
        n_encodings = 3
        encoding_size = 5
        model = Char2Vec(n_encodings=n_encodings, encoding_size=encoding_size)

        input_tensor = torch.zeros(batch_size, n_encodings)
        input_tensor[:, 0] = 1
        output_tensor = model(input_tensor)

        assert input_tensor.shape == output_tensor.shape

    @staticmethod
    def test_char2vec_with_mask():
        batch_size = 4
        n_encodings = 3
        encoding_size = 5
        model = Char2Vec(n_encodings=n_encodings, encoding_size=encoding_size)

        input_tensor = torch.zeros(batch_size, n_encodings)
        input_tensor[:, 0] = 1
        mask = np.array([True, True, False], dtype=bool)
        output_tensor = model(input_tensor, mask=mask)

        assert input_tensor[:, mask].shape == output_tensor.shape

    @pytest.mark.skip(reason="takes too long and saves encodings")
    @staticmethod
    def test_char2vec_usecase():
        char2vec(encoding_size=3, n_epochs=1)
