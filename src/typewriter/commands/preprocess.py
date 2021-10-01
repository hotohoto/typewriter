import click
from typewriter.usecases.preprocess import preprocess


@click.command()
def run_preprocess():
    preprocess()


if __name__ == "__main__":
    run_preprocess()
