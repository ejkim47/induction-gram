# Prepare Dataset for Training Fuzzy Matching Model and Evaluating Models

## Dataset for Training Fuzzy Matching Model
We provide two types of preprocessing code:
- For the OpenWebtext dataset, please refer to `data/openwebtext/readme.md`.
- For the Pile dataset, please refer to the following instructions to build a suffix array.

## Dataset for Evaluating Models
The evaluation dataset is built as a suffix array for Infini-Gram.
There are two options to build the suffix array: 'babylm', 'fineweb', and 'pile-uncopyrighted'.

### BabyLM
The data is available at this [website](https://osf.io/ad7qg/).   Download `text_data > test.zip` and unzip the file to create the test dataset. Make sure to replace `DATASET_ROOT` with the path to the unzipped directory when running the code.

### FineWeb
The dataset is automatically downloaded through [huggingface repo](https://huggingface.co/datasets/HuggingFaceFW/fineweb).
For the test set, we use 'sample-10BT'.

### Pile-uncopyrighted
The dataset is automatically downloaded from [huggingface repo](https://huggingface.co/datasets/monology/pile-uncopyrighted).
This dataset is only used to train a fuzzy matching model.

### Running the Code
To build the suffix array, run the following command:
```bash
python 00_prepare_data.py --dataset_name DATASET_NAME --data_dir DATASET_ROOT --save_dir SAVE_PATH --tokenizer TOKENIZER --batch_size BATCH_SIZE --mem MEMORY
```
Set `SAVE_PATH` as a path to a directory named `{DATASET_NAME}_{TOKENIZER}` inside `alm.config.DATA_DIR_ROOT`.


This code is modified from the official code to build Infini-Gram.
Detailed explanations can be found on this [website](https://infini-gram.io/pkg_doc.html).