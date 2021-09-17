import click
from typewriter.usecases.char2vec import build_embeddings, load_embeddings, save_embeddings


@click.command()
def run_char2vec():
    embeddings = load_embeddings()
    if embeddings is None:
        print("Building new embeddings...")
        embeddings = build_embeddings()
        save_embeddings(embeddings)
    else:
        print("Updating the existing embeddings...")
        build_embeddings(embeddings=embeddings)
