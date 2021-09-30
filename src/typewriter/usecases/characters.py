import json
import pathlib

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
    characters = CONTROL_CHARACTERS.copy()
    input_dir = pathlib.Path("data/1_preprocessed")
    file_paths = input_dir.iterdir()
    for file_path in file_paths:
        with open(file_path) as f:
            characters |= set(f.read())
    return sorted(characters)


def load_characters():
    if not PATH_TO_CHARACTERS.is_file():
        return None

    with open(PATH_TO_CHARACTERS) as f:
        return json.loads(f.read())


def save_characters(characters):
    assert characters
    assert isinstance(characters, list)
    for c in characters:
        assert isinstance(c, str)
        assert c

    with open(PATH_TO_CHARACTERS, "w") as f:
        f.write(json.dumps(characters))
