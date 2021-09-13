import pathlib

from torch.utils.data import DataLoader, Dataset
from typewriter.transforms.compose import Compose
from typewriter.transforms.mark_end import MarkEnd
from typewriter.transforms.onehot import OneHot
from typewriter.transforms.skip_gram import SkipGram
from typewriter.usecases.characters import get_characters


def train_generator():
    characters = get_characters()
    data_loader = DataLoader(
        TextDataset(
            transform=Compose(
                [
                    MarkEnd(),
                    OneHot(characters),
                    SkipGram(left=3, right=3),
                ]
            )
        ),
        batch_size=1,
        shuffle=True,
    )

    for batch in data_loader:
        print(batch)
        print(batch.shape)
        break


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
