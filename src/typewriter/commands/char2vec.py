import click
from typewriter.usecases.char2vec import train_embeddings, load_embeddings, save_embeddings


@click.command()
def run_char2vec():
    embeddings = load_embeddings()
    if embeddings is None:
        print("Training new embeddings...")
        embeddings = train_embeddings()
        save_embeddings(embeddings)
    else:
        print("Updating the existing embeddings...")
        train_embeddings(embeddings=embeddings)


if __name__ == "__main__":
    run_char2vec()
