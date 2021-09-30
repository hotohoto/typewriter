import json
import pathlib

from torch.utils.data import DataLoader, Dataset
from typewriter.transforms.character_to_embeddings import CharacterToEmbedding
from typewriter.transforms.compose import Compose
from typewriter.transforms.mark_end import MarkEnd
from typewriter.usecases.char2vec import get_embeddings

PATH_TO_GENERATOR = pathlib.Path("data/3_models/generator.json")


def get_generator():
    generator = load_generator()
    if generator:
        return generator

    generator = train()
    save_generator(generator)

    return generator


def train():
    embeddings = get_embeddings()
    data_loader = DataLoader(
        TextDataset(transform=Compose([MarkEnd(), CharacterToEmbedding(embeddings)])),
        batch_size=1,
        shuffle=True,
    )

    for batch in data_loader:
        print(batch)
        print(batch.shape)
        break

    return generator


def load_generator():
    if not PATH_TO_GENERATOR.is_file():
        return None

    with open(PATH_TO_GENERATOR) as f:
        return json.loads(f.read())


def save_generator(generator):
    assert generator
    assert isinstance(generator, dict)
    for c in generator:
        assert isinstance(c, str)
        assert c

    with open(PATH_TO_GENERATOR, "w") as f:
        f.write(json.dumps(generator))


class TextDataset(Dataset):
    def __init__(self, base_dir=None, transform=None):
        super().__init__()
        base_dir = pathlib(base_dir) if base_dir else pathlib.Path("data/1_preprocessed")
        assert base_dir.is_dir()
        self.base_dir = base_dir

        files = [f for f in base_dir.iterdir() if f.is_file()]
        assert files
        self.files = files

        texts = []

        for f in files:
            with open(f) as fin:
                texts.append(fin.read())

        self.texts = texts
        self.transform = transform

    def __getitem__(self, key):

        text = self.texts[key]

        if self.transform:
            text = self.transform(text)

        return text

    def __len__(self):
        return len(self.texts)
