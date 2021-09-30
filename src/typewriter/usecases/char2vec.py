import json
import math
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from typewriter.datasets.skip_gram_dataset import SkipGramDataset
from typewriter.usecases.characters import get_characters
from typewriter.values.embeddings import Embeddings

PATH_TO_EMBEDDINGS = pathlib.Path("data/2_characters/embeddings.json")


def get_embeddings():
    embeddings = load_embeddings()
    if embeddings:
        return embeddings

    embeddings = train_embeddings()
    save_embeddings(embeddings)

    return embeddings


def train_embeddings(encoding_size=16, n_epochs=1, embeddings: Embeddings = None):
    characters = get_characters()
    data_loader = DataLoader(
        SkipGramDataset(
            characters=characters,
            window=2,
        ),
        batch_size=64,
        shuffle=True,
    )

    if embeddings:
        w_in = embeddings.w_in
        w_out = embeddings.w_out
        model = Char2Vec(len(characters), encoding_size, w_in=w_in, w_out=w_out)
    else:
        model = Char2Vec(len(characters), encoding_size)

    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())

    recent_loss_values = []

    for epoch in range(n_epochs):
        print(f"epoch = {epoch}")
        for i, (text, context) in enumerate(data_loader):
            mask = (text + context).sum(axis=0).bool()
            # TODO append negative samples to mask
            prediction = model(text, mask)
            loss = criterion(prediction, context[:, mask])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            recent_loss_values.append(loss.item())

            if i % 1000 == 0:
                print(f"i={i}, mean loss={np.mean(recent_loss_values)}")
                recent_loss_values = []
    return Embeddings(
        characters=characters,
        w_in=model.w_in.detach().numpy(),
        w_out=model.w_out.detach().numpy(),
    )


def load_embeddings():
    if not PATH_TO_EMBEDDINGS.is_file():
        return None

    with open(PATH_TO_EMBEDDINGS) as f:
        return Embeddings.from_dict(json.loads(f.read()))


def save_embeddings(embeddings):
    assert embeddings
    assert isinstance(embeddings, Embeddings)
    for c in embeddings:
        assert isinstance(c, str)
        assert c

    with open(PATH_TO_EMBEDDINGS, "w") as f:
        f.write(json.dumps(embeddings.to_dict()))


class Char2Vec(torch.nn.Module):
    def __init__(
        self, n_embeddings: int, encoding_size: int, w_in=None, w_out=None, device=None, dtype=None
    ):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.n_embeddings = n_embeddings
        self.encoding_size = encoding_size

        if w_in is not None:
            assert len(w_in) == encoding_size
            self.w_in = torch.nn.Parameter(torch.tensor(w_in, **factory_kwargs))
        else:
            self.w_in = torch.nn.Parameter(
                torch.empty((encoding_size, n_embeddings), **factory_kwargs)
            )
            torch.nn.init.kaiming_uniform_(self.w_in, a=math.sqrt(5))

        if w_out is not None:
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
