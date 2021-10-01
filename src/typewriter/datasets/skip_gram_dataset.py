import pathlib
from functools import lru_cache

from torch.utils.data.dataset import Dataset
from typewriter.transforms.compose import Compose
from typewriter.transforms.mark_end import MarkEnd
from typewriter.transforms.onehot import OneHot
from typewriter.transforms.skip_gram import SkipGram
from typewriter.values.characters import Characters


class SkipGramDataset(Dataset):
    def __init__(self, base_dir=None, characters: Characters = None, window=1):
        super().__init__()

        base_dir = pathlib(base_dir) if base_dir else pathlib.Path("data/1_preprocessed")
        assert base_dir.is_dir()
        self.base_dir = base_dir

        file_paths = [f for f in base_dir.iterdir() if f.is_file()]
        assert file_paths
        self.file_paths = file_paths

        skip_gram = SkipGram(window=window)
        self.transform = Compose(
            [
                MarkEnd(),
                OneHot(characters),
                skip_gram,
            ]
        )

        text_end_indices = []  # exclusive
        prev_end = 0
        for p in file_paths:
            with open(p) as f:
                length = skip_gram.output_length(len(f.read()))
                end = prev_end + length
                text_end_indices.append(end)
                prev_end = end

        self.text_end_indices = text_end_indices

    @lru_cache(maxsize=1024 * 3)
    def _get_transformed_text(self, path):
        with open(path) as f:
            return self.transform(f.read())

    def _find_file(self, idx_to_find):
        for i, current in enumerate(self.text_end_indices):
            if idx_to_find < current:
                offset = idx_to_find - (self.text_end_indices[i - 1] if i > 0 else 0)
                return self.file_paths[i], offset
        raise IndexError(f"idx_to_find={idx_to_find}")

    def __getitem__(self, index):
        path, offset = self._find_file(index)
        context, text = self._get_transformed_text(path)
        return context[offset], text[offset]

    def __len__(self):
        return self.text_end_indices[-1]
