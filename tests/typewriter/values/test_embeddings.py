import numpy as np
from typewriter.values.embeddings import Embeddings


class TestEmbeddings:
    @staticmethod
    def test_closest():
        w_in = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]).T.tolist()
        embeddings = Embeddings.from_dict(
            {
                "characters": ["1", "2", "3", "4"],
                "w_in": w_in,
                "w_out": np.random.rand(3, 4).tolist(),
            }
        )
        result = embeddings.closest(query_embeddings=np.array([1.1, 1.6, 0.8]))
        assert result == np.array("1")

        result = embeddings.closest(query_embeddings=np.array([2.1, 2.6, 1.8]))
        assert result == np.array("2")

        result = embeddings.closest(query_embeddings=np.array([3.1, 3.6, 2.8]))
        assert result == np.array("3")

        result = embeddings.closest(
            query_embeddings=np.array([[3.1, 3.6, 2.8], [2.1, 2.6, 1.8], [4.1, 4.6, 3.8]])
        )
        assert (result == np.array(["3", "2", "4"])).all()

    @staticmethod
    def test_encode():
        w_in = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]).T.tolist()
        embeddings = Embeddings.from_dict(
            {
                "characters": ["1", "2", "3", "4"],
                "w_in": w_in,
                "w_out": np.random.rand(3, 4).tolist(),
            }
        )
        w_in = np.array(w_in)
        result = embeddings.encode(text="1")
        assert (result == w_in[:, 0].T).all()

        result = embeddings.encode(text="2")
        assert (result == w_in[:, 1].T).all()

        result = embeddings.encode(text="3")
        assert (result == w_in[:, 2].T).all()

        result = embeddings.encode(text="324")
        assert (result == w_in[:, [2, 1, 3]].T).all()
