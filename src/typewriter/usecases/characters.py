import json
import os

PATH_TO_CHARACTERS = os.path.join("data", "2_characters", "characters.json")


def get_characters():
    characters = load_characters()
    if characters:
        return characters

    characters = build_characters()
    save_characters(characters)

    return characters


def build_characters():
    characters = set()
    input_dir = os.path.join("data", "1_preprocessed")
    files = os.listdir(input_dir)
    file_paths = [os.path.join(input_dir, f) for f in files]
    for file_path in file_paths:
        with open(file_path) as f:
            characters |= set(f.read())
    return sorted(characters)


def load_characters():
    if not os.path.isfile(PATH_TO_CHARACTERS):
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
