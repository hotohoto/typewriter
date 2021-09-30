# A character level text generator

(toy project)

## Features

- preprocessor
  - decompose Korean
- postprocessor
  - reconstruct Korean
- naive char2vec mimicking word2vec
- seq2seq text generation
- ml algorithsm candidates
  - RNN: LSTM or GRU
    - character level
      - encoder
      - decoder
      - attention
    - word level
    - sentence level
    - document level
    - https://github.com/sjchoi86/Tensorflow-101/blob/master/notebooks/char_rnn_train_hangul.ipynb
      - gradient clipping
      - multi layer lstm
      - initial cell state is zero
      - predict next character
      - cell state burning for inference
    - seq2seq
      - we can just compare the 2 sequences with paddings
  - RNN + attention
  - hugging face model

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

(papers)
(2020)
- [Character-level Transformer-based Neural Machine Translation](https://arxiv.org/abs/2005.11239)
- [Character-Level Translation with Self-attention](https://arxiv.org/abs/2004.14788)
  - https://github.com/CharizardAcademy/convtransformer

(2018)
- [Character-Level Language Modeling with Deeper Self-Attention](https://arxiv.org/abs/1808.04444)

(2017)
- [Fully Character-Level Neural Machine Translation without Explicit Segmentation](https://arxiv.org/abs/1610.03017)
  - https://github.com/nyu-dl/dl4mt-c2c

(blog)
- https://towardsdatascience.com/besides-word-embedding-why-you-need-to-know-character-embedding-6096a34a3b10
  - https://github.com/makcedward/nlp/blob/master/sample/nlp-character_embedding.ipynb

(datasets)
- https://github.com/jihunkim625/201500844_Ji-Hun-Kim_Korean-song-lyrics-analysis

(libraries)
- https://github.com/bluedisk/hangul-toolkit
- https://github.com/pytorch/fairseq

(etc)
- https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
- https://github.com/hotohoto/python-example-project
