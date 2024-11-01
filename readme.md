# Interpretable Language Modeling via Induction-head Ngram Models

This is official code repository for 'Interpretable Language Modeling via Induction-head Ngram Models'.

### Setup
- Clone the repo and run `pip install -e .` to install the `alm` package locally
- Set paths/env variables in `alm/config.py` to point to the correct directories to store data

### Experiments
- For language modeling experiments, please refer to `experiments/readme.md`.


### Organization
- `alm` is the main package
  - `alm/config.py` is the configuration file that points to where things should be stored
  - `alm/data` contains data utilities
  - `alm/models` contains source for models
- `data` contains code for preprocessing data
- `experiments` contains code for language modeling experiments
