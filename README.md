# A character level text generator

(toy project)

## Features

- dataset downloader
- preprocessor
  - decompose Korean
- postprocessor
  - reconstruct Korean
- naive char2vec mimicking word2vec
- seq2seq text generation
- ml algorithsm candidates
  - RNN: LSTM or GRU
  - RNN + attention
  - huging face model

## Development environment setup

### Setup virtual environment

With the `venv` module:

```bash
python3 -m venv venv
source venv/bin/activate
```

With the `pyenv`

```bash
pyenv virtualenv 3.8.3 my-venv
pyenv local my-venv
```

### Do further setup

```bash
pip install -U pip
poetry install
poetry hello
```

### Run tests

```bash
pytest
```

### Run scripts

```bash
poetry run hello
python src/main.py
```

### Example VS Code configuration

`settings.json`:

```json
{
    "python.linting.enabled": true,
    "python.linting.flake8Enabled": true,
    "python.linting.pylintEnabled": false,
    "python.testing.nosetestsEnabled": false,
    "python.testing.pytestArgs": [
        "tests"
    ],
    "python.testing.pytestEnabled": true,
    "python.testing.unittestEnabled": false
}
```


## References

(datasets)
- https://github.com/jihunkim625/201500844_Ji-Hun-Kim_Korean-song-lyrics-analysis

(libraries)
- https://github.com/bluedisk/hangul-toolkit

(etc)
- https://github.com/hotohoto/python-example-project
