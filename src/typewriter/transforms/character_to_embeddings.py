from typewriter.values.embeddings import Embeddings


class CharacterToEmbedding(object):
    def __init__(self, embeddings: Embeddings):
        assert embeddings
        self.embeddings = embeddings.copy()

    def __call__(self, text):
        return self.embeddings.encode(text)
