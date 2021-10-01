import click
from typewriter.usecases.train import load_generator, save_generator, train


@click.command()
def run_train():
    train()
    generator = load_generator()
    if generator is None:
        print("Training new generator...")
        generator = train()
        save_generator(generator)
    else:
        print("Updating the existing generator...")
        train(generator=generator)


if __name__ == "__main__":
    run_train()
