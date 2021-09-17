import json
import math
import os
import pathlib

import numpy as np
import torch
from torch.utils.data import DataLoader
from typewriter.datasets.skip_gram_dataset import SkipGramDataset
from typewriter.usecases.characters import get_characters

PATH_TO_ENCODINGS = pathlib.Path("data/2_characters/embeddings.json")


def get_encodings():
    encodings = load_encodings()
    if encodings:
        return encodings

    encodings = build_encodings()
    save_encodings(encodings)

    return encodings


def build_encodings(encoding_size=16, n_epochs=1, encodings=None):
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

    if encodings:
        w_in = encodings["w_in"]
        w_out = encodings["w_out"]
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


def load_encodings():
    if not os.path.isfile(PATH_TO_ENCODINGS):
        return None

    with open(PATH_TO_ENCODINGS) as f:
        return json.loads(f.read())


def save_encodings(encodings):
    assert encodings
    assert isinstance(encodings, dict)
    for c in encodings:
        assert isinstance(c, str)
        assert c

    with open(PATH_TO_ENCODINGS, "w") as f:
        f.write(json.dumps(encodings))


class Char2Vec(torch.nn.Module):
    def __init__(self, n_encodings, encoding_size, w_in=None, w_out=None, device=None, dtype=None):
        super().__init__()
        factory_kwargs = {"device": device, "dtype": dtype}
        self.n_encodings = n_encodings
        self.encoding_size = encoding_size

        if w_in:
            assert len(w_in) == encoding_size
            self.w_in = torch.nn.Parameter(torch.tensor(w_in, **factory_kwargs))
        else:
            torch.nn.Parameter(torch.empty((encoding_size, n_encodings), **factory_kwargs))
            torch.nn.init.kaiming_uniform_(self.w_in, a=math.sqrt(5))

        if w_out:
            assert len(w_out) == n_encodings
            self.w_out = torch.nn.Parameter(torch.tensor(w_out, **factory_kwargs))
        else:
            torch.nn.Parameter(torch.empty((n_encodings, encoding_size), **factory_kwargs))
            torch.nn.init.kaiming_uniform_(self.w_out, a=math.sqrt(5))

    def forward(self, x, mask=None):
        if mask is None:
            x = self.w_in @ x.T
            x = self.w_out @ x
        else:
            x = self.w_in[:, mask] @ x[:, mask].T
            x = self.w_out[mask, :] @ x
        return x.T
