## OpenWebText Dataset

After running `00_prepare.py` (preprocessing), we obtain the following:

- `train.bin` is approximately 17GB, and `val.bin` is about 8.5MB.
- The training set contains around 9B tokens.
- The validation set contains around 4M tokens.

These tokens were extracted from a total of 8,013,769 documents using the GPT-2 tokenizer.


References:

- OpenAI's WebText dataset is discussed in [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [OpenWebText](https://skylion007.github.io/OpenWebTextCorpus/) dataset
