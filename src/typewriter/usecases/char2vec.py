from torch.utils.data import DataLoader
from typewriter.datasets.skip_gram_dataset import SkipGramDataset
from typewriter.usecases.characters import get_characters


def char2vec():
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

    for batch in data_loader:
        # TODO train char2vec encoder
        print(batch)
        print(batch.shape)
        break
