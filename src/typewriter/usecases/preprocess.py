import pathlib
import random
from typewriter.transforms.hangul import DecomposeHangul


def preprocess():
    random.seed(0)

    original_data_path = pathlib.Path("data/0_original")
    src_dirs = original_data_path.iterdir()
    output_dir = pathlib.Path("data/1_preprocessed")
    output_dir.mkdir(parents=True, exist_ok=True)

    all_source_paths = []

    for src_dir in src_dirs:
        if not src_dir.is_dir():
            continue
        all_source_paths += [p for p in src_dir.iterdir() if p.suffix == ".txt"]

    digit = len(str(len(all_source_paths)))
    transform = DecomposeHangul()
    for idx, src_path in enumerate(all_source_paths):
        output_path = output_dir.joinpath(f"{str(idx).zfill(digit)}.txt")
        with open(src_path) as fin, open(output_path, "w") as fout:
            fout.write(transform(fin.read().strip() + "\n"))
