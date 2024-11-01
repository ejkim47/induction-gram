# Mech LM

### Setup
- Download pre-built indexes from [Infini-gram Project Page](https://infini-gram.io/pkg_doc.html) and set path `alm/config.py` to point to the correct directories to store indexes



## Prepare Dataset
Please refer to data/ to prepare datasets for training fuzzy matchin model and evaluating models.

## Train Fuzzy
Code example to train fuzzy matching model with 4 gpus:
```
torchrun --standalone --nproc_per_node=4 experiments/01_train_simlm.py --learning_rate 0.0001 --sampling_size 128 --sampling_size2 512 --batch_size 32 --n_layer 4 --n_out_embd 128 --similarity_function cosine --context_length 32 --n_head 8 --lamb_kl 1.0 --temperature 0.1 --attention_block ModifiedTransformerBlock --teacher_llm llama2-7B
```


## Building Infini-Gram
Please refer to [Infini-gram Project Page](https://infini-gram.io/pkg_doc.html) to build Infini-Gram.
Save local indexes in `INFINIGRAM_INDEX_PATH` (set in `alm/config.py`).


## Evaluation
Code example to evaluate Induction-Gram
```
python experiments/02_eval.py --model_type MODEL_TYPE --dataset DATASET --tokenizer_checkpoint TOKENIZER --checkpoint INFINIGRAM_INDEX/LLM --analyze True --fuzzy_checkpoint FUZZY_MATCHING_MODEL_PATH
```


## Note
