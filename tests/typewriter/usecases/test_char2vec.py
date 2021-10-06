import pytest
import torch
from typewriter.usecases.char2vec import Char2Vec, train_embeddings


class TestChar2Vec:
    @staticmethod
    def test_char2vec():
        n_embeddings = 4
        encoding_size = 5
        model = Char2Vec(n_embeddings=n_embeddings, encoding_size=encoding_size)

        text_indices = torch.tensor([0, 1, 1, 2, 2, 3], dtype=int)
        context_indices = torch.tensor([1, 0, 2, 1, 3, 2], dtype=int)
        logits = model(text_indices, context_indices)

        assert context_indices.shape == logits.shape

    @pytest.mark.skip(reason="takes too long and updates embeddings saved")
    @staticmethod
    def test_train_embeddings():
        train_embeddings(encoding_size=3, n_epochs=1)
