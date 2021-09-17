import typing as t

import numpy as np


class Embeddings:
    def __init__(self, characters: t.List[str], w_in: np.ndarray, w_out: np.ndarray):
        self.characters = characters.copy()
        self.w_in = np.array(w_in)
        self.w_out = np.array(w_out)

    def closest(self, query_embeddings) -> np.ndarray:
        assert query_embeddings is not None
        query_embeddings = (
            query_embeddings
            if isinstance(query_embeddings, np.ndarray)
            else np.array(query_embeddings)
        )
        embedding_size = self.w_in.shape[0]
        assert query_embeddings.shape[-1] == embedding_size

        target_shape = query_embeddings.shape[:-1]
        query_embeddings = query_embeddings.reshape(-1, embedding_size)

        results = []
        for e in query_embeddings:
            diff = np.array(self.w_in).T - e
            idx = np.argmin((diff ** 2).sum(axis=1), axis=0)
            results.append(self.characters[idx])

        return np.array(results).reshape(target_shape)

    def encode(self, text) -> np.ndarray:
        mapping = {c: i for i, c in enumerate(self.characters)}
        indices = [mapping[c] for c in text]
        w_in = np.array(self.w_in)
        return w_in[:, indices].T

    def to_dict(self):
        return {
            "characters": self.characters,
            "w_in": self.w_in.tolist(),
            "w_out": self.w_out.tolist(),
        }

    def __eq__(self, o: object) -> bool:
        return isinstance(o, Embeddings) and self.to_dict() == o.to_dict()

    def copy(self):
        return Embeddings.from_dict(self.to_dict())

    @staticmethod
    def from_dict(data):
        return Embeddings(
            characters=data["characters"],
            w_in=data["w_in"],
            w_out=data["w_out"],
        )
