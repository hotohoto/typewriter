import numpy as np
import pytest
import torch
from typewriter.usecases.char2vec import Char2Vec, build_embeddings, closest


class TestChar2Vec:
    @staticmethod
    def test_char2vec():

        batch_size = 4
        n_embeddings = 3
        encoding_size = 5
        model = Char2Vec(n_embeddings=n_embeddings, encoding_size=encoding_size)

        input_tensor = torch.zeros(batch_size, n_embeddings)
        input_tensor[:, 0] = 1
        output_tensor = model(input_tensor)

        assert input_tensor.shape == output_tensor.shape

    @staticmethod
    def test_char2vec_with_mask():
        batch_size = 4
        n_embeddings = 3
        encoding_size = 5
        model = Char2Vec(n_embeddings=n_embeddings, encoding_size=encoding_size)

        input_tensor = torch.zeros(batch_size, n_embeddings)
        input_tensor[:, 0] = 1
        mask = np.array([True, True, False], dtype=bool)
        output_tensor = model(input_tensor, mask=mask)

        assert input_tensor[:, mask].shape == output_tensor.shape

    @staticmethod
    def test_closest():
        w_in = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]).T.tolist()
        embeddings = {
            "characters": ["1", "2", "3", "4"],
            "w_in": w_in,
            "w_out": np.random.rand(3, 4).tolist(),
        }
        result = closest(query_embeddings=np.array([1.1, 1.6, 0.8]), embeddings=embeddings)
        assert result == np.array("1")

        result = closest(query_embeddings=np.array([2.1, 2.6, 1.8]), embeddings=embeddings)
        assert result == np.array("2")

        result = closest(query_embeddings=np.array([3.1, 3.6, 2.8]), embeddings=embeddings)
        assert result == np.array("3")

        result = closest(
            query_embeddings=np.array([[3.1, 3.6, 2.8], [2.1, 2.6, 1.8], [4.1, 4.6, 3.8]]),
            embeddings=embeddings,
        )
        assert (result == np.array(["3", "2", "4"])).all()

    @pytest.mark.skip(reason="takes too long and updates embeddings saved")
    @staticmethod
    def test_build_embeddings():
        build_embeddings(encoding_size=3, n_epochs=1)
