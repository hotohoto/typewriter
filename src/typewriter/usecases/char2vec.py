import json
import math
import pathlib

import numpy as np
import runstats
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


def train_embeddings(encoding_size=16, n_epochs=1, embeddings: Embeddings = None, t=0.1):
    np.random.seed(0)
    torch.manual_seed(0)

    characters = get_characters()

    batch_size = 64
    data_loader = DataLoader(
        SkipGramDataset(
            characters=characters,
            window=2,
        ),
        batch_size=batch_size,
        shuffle=True,
    )

    if embeddings:
        w_in = embeddings.w_in
        w_out = embeddings.w_out
        model = Char2Vec(len(characters), encoding_size, w_in=w_in, w_out=w_out)
    else:
        model = Char2Vec(len(characters), encoding_size)

    def criterion(logit: torch.Tensor, target: torch.Tensor):
        return -(
            target * torch.log(torch.sigmoid(logit))
            + (1 - target) * torch.log(torch.sigmoid(-logit))
        ).mean()

    optimizer = torch.optim.Adam(model.parameters())

    stats_loss = runstats.Statistics()
    stats_n_discarded_rows = runstats.Statistics()
    stats_n_negative_samples = runstats.Statistics()

    character_list = characters.list()

    word_probs = np.array([characters.probs[c] for c in character_list])

    negative_sampling_probs = word_probs ** (3 / 4)
    negative_sampling_probs *= negative_sampling_probs.sum()

    # probs to discard samples for each character
    probs_to_discard = 1 - (t / (word_probs + np.finfo(float).eps)) ** 0.5
    probs_to_discard[probs_to_discard < 0] = 0

    for epoch in range(n_epochs):
        print(f"epoch = {epoch}")
        for i, (text, context) in enumerate(data_loader):
            batch_size = len(text)

            # subsampling
            rows_to_discard = torch.zeros(batch_size, dtype=bool)
            character_indices_to_discard = np.where(np.random.binomial(1, p=probs_to_discard))[0]
            for idx in character_indices_to_discard:
                rows_to_discard |= (text == idx).any()
                rows_to_discard |= (context == idx).any()
            stats_n_discarded_rows.push(rows_to_discard.sum().item())

            if not rows_to_discard.any():
                text = text[~rows_to_discard]
                context = context[~rows_to_discard]

                # negative sampling
                all_contexts = [context]
                all_texts = [text]
                negative_sampling_indices = np.where(
                    np.random.binomial(1, p=negative_sampling_probs)
                )[0]
                for idx in negative_sampling_indices:
                    neg_ctx = torch.full(context.shape, idx)
                    neg_txt = text
                    # mask = neg_ctx != context
                    # neg_ctx = neg_ctx[mask]
                    # neg_txt = text[mask]
                    all_contexts.append(neg_ctx)
                    all_texts.append(neg_txt)
                combined_context = torch.cat(all_contexts)
                combined_text = torch.cat(all_texts)
                combined_targets = torch.cat(
                    (torch.ones(context.shape), torch.zeros(len(combined_context) - len(context)))
                )
                stats_n_negative_samples.push(len(negative_sampling_indices))

                # train
                logit = model(combined_text, combined_context)
                loss = criterion(logit, combined_targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                stats_loss.push(loss.item())

            if i % 1000 == 0:
                print(
                    f"i={i}, "
                    f"mean(loss)={stats_loss.mean()}, "
                    f"mean(# discarded rows)={stats_n_discarded_rows.mean()}, "
                    f"mean(# negative samples)={stats_n_negative_samples.mean()}"
                )
                stats_loss.clear()
                stats_n_discarded_rows.clear()
                stats_n_negative_samples.clear()

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


def save_embeddings(embeddings: Embeddings):
    assert embeddings
    assert isinstance(embeddings, Embeddings)
    for c in embeddings.characters.keys:
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
                torch.empty((n_embeddings, encoding_size), **factory_kwargs)
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

    def forward(self, text_indices, context_indices):
        # text_indices.shape = (batch_size,)
        # context_indices.shape = (batch_size,)

        x = self.w_in[text_indices, :]

        # x.shape = (batch_size, encoding_size)

        x = (x * self.w_out[context_indices, :]).sum(dim=1)

        # x.shape = (batch_size, )

        return x
