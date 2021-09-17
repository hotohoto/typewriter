import click
from typewriter.usecases.char2vec import build_encodings, load_encodings, save_encodings


@click.command()
def run_char2vec():
    encodings = load_encodings()
    if encodings is None:
        print("Building new encodings...")
        encodings = build_encodings()
        save_encodings(encodings)
    else:
        print("Updating the existing encodings...")
        build_encodings(encodings=encodings)
