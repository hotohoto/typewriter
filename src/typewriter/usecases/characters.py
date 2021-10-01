import json
import pathlib
from collections import Counter

from typewriter.values.characters import Characters
from typewriter.values.values import CONTROL_CHARACTERS

PATH_TO_CHARACTERS = pathlib.Path("data/2_characters/characters.json")


def get_characters():
    characters = load_characters()
    if characters:
        return characters

    characters = build_characters()
    save_characters(characters)

    return characters


def build_characters():
    counter = Counter({c: 0 for c in CONTROL_CHARACTERS})
    input_dir = pathlib.Path("data/1_preprocessed")
    file_paths = input_dir.iterdir()
    for file_path in file_paths:
        with open(file_path) as f:
            counter.update(f.read())

    total = sum(counter.values())
    return Characters({char: count / total for char, count in counter.items()})


def load_characters():
    if not PATH_TO_CHARACTERS.is_file():
        return None

    with open(PATH_TO_CHARACTERS) as f:
        return Characters.from_dict(json.loads(f.read()))


def save_characters(characters):
    assert isinstance(characters, Characters)
    with open(PATH_TO_CHARACTERS, "w") as f:
        f.write(json.dumps(characters.to_dict()))
