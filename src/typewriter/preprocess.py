import click
import os
import random
from typewriter.utils.hangul import encode


@click.command()
def preprocess():
    random.seed(0)

    original_data_path = os.path.join("data", "0_original")
    src_dirs = os.listdir(original_data_path)
    output_dir = os.path.join("data", "1_preprocessed")
    os.makedirs(output_dir, exist_ok=True)

    all_source_paths = []

    for src_dir in src_dirs:
        dir_path = os.path.join(original_data_path, src_dir)
        if not os.path.isdir(dir_path):
            continue
        all_source_paths += [
            os.path.join(dir_path, f) for f in os.listdir(dir_path) if f.endswith(".txt")
        ]

    digit = len(str(len(all_source_paths)))
    for idx, src_path in enumerate(all_source_paths):
        output_path = os.path.join(output_dir, f"{str(idx).zfill(digit)}.txt")
        with open(src_path) as fin, open(output_path, "w") as fout:
            fout.write(encode(fin.read()))
