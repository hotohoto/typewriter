import json
import math
import os
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from typewriter.datasets.skip_gram_dataset import SkipGramDataset
from typewriter.usecases.characters import get_characters

PATH_TO_EMBEDDINGS = pathlib.Path("data/2_characters/embeddings.json")


def get_embeddings():
    embeddings = load_embeddings()
    if embeddings:
        return embeddings

    embeddings = train_embeddings()
    save_embeddings(embeddings)

    return embeddings


def train_embeddings(encoding_size=16, n_epochs=1, embeddings=None):
    characters = get_characters()
    data_loader = DataLoader(
        SkipGramDataset(
            characters=characters,
            left=2,
            skip=1,
            right=2,
        ),
        batch_size=64,
        shuffle=True,
    )

    if embeddings:
        w_in = embeddings["w_in"]
        w_out = embeddings["w_out"]
        model = Char2Vec(len(characters), encoding_size, w_in=w_in, w_out=w_out)
    else:
        model = Char2Vec(len(characters), encoding_size)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    recent_loss_values = []

    for epoch in range(n_epochs):
        print(f"epoch = {epoch}")
        for i, (context, text) in enumerate(data_loader):
            mask = (context + text).sum(axis=0).bool()
            prediction = model(text, mask)
            loss = criterion(prediction, context[:, mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            recent_loss_values.append(loss.item())

            if i % 1000 == 0:
                print(f"i={i}, mean loss={np.mean(recent_loss_values)}")
                recent_loss_values = []
    return {
        "characters": characters,
        "w_in": model.w_in.detach().numpy().tolist(),
        "w_out": model.w_out.detach().numpy().tolist(),
    }


def load_embeddings():
    if not os.path.isfile(PATH_TO_EMBEDDINGS):
        return None

    with open(PATH_TO_EMBEDDINGS) as f:
        return json.loads(f.read())


def save_embeddings(embeddings):
    assert embeddings
    assert isinstance(embeddings, dict)
    for c in embeddings:
        assert isinstance(c, str)
        assert c

    with open(PATH_TO_EMBEDDINGS, "w") as f:
        f.write(json.dumps(embeddings))


def closest(query_embeddings, embeddings: dict) -> np.ndarray:
    characters = embeddings["characters"]
    w_in = np.array(embeddings["w_in"])
    assert query_embeddings is not None
    query_embeddings = (
        query_embeddings if isinstance(query_embeddings, np.ndarray) else np.array(query_embeddings)
    )
    embedding_size = w_in.shape[0]
    assert query_embeddings.shape[-1] == embedding_size

    target_shape = query_embeddings.shape[:-1]
    query_embeddings = query_embeddings.reshape(-1, embedding_size)

    results = []
    for e in query_embeddings:
        diff = np.array(embeddings["w_in"]).T - e
        idx = np.argmin((diff ** 2).sum(axis=1), axis=0)
        results.append(characters[idx])

    return np.array(results).reshape(target_shape)


def encode(text, embeddings: dict) -> np.ndarray:
    characters = embeddings["characters"]
    mapping = {c: i for i, c in enumerate(characters)}
    indices = [mapping[c] for c in text]
    w_in = np.array(embeddings["w_in"])
    return w_in[:, indices].T


class Char2Vec(torch.nn.Module):
    def __init__(self, n_embeddings, encoding_size, w_in=None, w_out=None, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.n_embeddings = n_embeddings
        self.encoding_size = encoding_size

        if w_in:
            assert len(w_in) == encoding_size
            self.w_in = torch.nn.Parameter(torch.tensor(w_in, **factory_kwargs))
        else:
            self.w_in = torch.nn.Parameter(
                torch.empty((encoding_size, n_embeddings), **factory_kwargs)
            )
            torch.nn.init.kaiming_uniform_(self.w_in, a=math.sqrt(5))

        if w_out:
            assert len(w_out) == n_embeddings
            self.w_out = torch.nn.Parameter(torch.tensor(w_out, **factory_kwargs))
        else:
            self.w_out = torch.nn.Parameter(
                torch.empty((n_embeddings, encoding_size), **factory_kwargs)
            )
            torch.nn.init.kaiming_uniform_(self.w_out, a=math.sqrt(5))

    def forward(self, x, mask=None):
        if mask is None:
            x = self.w_in @ x.T
            x = self.w_out @ x
        else:
            x = self.w_in[:, mask] @ x[:, mask].T
            x = self.w_out[mask, :] @ x
        return x.T
