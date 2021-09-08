import click
from typewriter.usecases.char2vec import char2vec


@click.command()
def run_char2vec():
    char2vec()
