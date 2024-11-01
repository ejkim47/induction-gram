import os
import alm.data.dsets
from collections import defaultdict
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from os.path import join
import logging
import numpy as np
import random
import torch

import argparse
import pandas as pd
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import Subset
import alm.config
import alm.data.dsets
from alm.models import InfiniGramMulti, InductionOnly, InductionGram

LLM_MAP = {'llama2-7B': 'meta-llama/Llama-2-7b-hf',
           'llama3-8B': 'meta-llama/Meta-Llama-3-8B',
           'llama2': 'meta-llama/Llama-2-7b-hf',
           'llama3': 'meta-llama/Meta-Llama-3-8B'}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False, help="if debug")
    parser.add_argument("--analyze", type=bool, default=False, help="if save prediction for analysis")
    parser.add_argument("--resume", type=bool, default=False, help="if resume from previous prediction")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(alm.config.SAVE_DIR_ROOT, "mechlm", "eval"),
        help="directory for saving",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default=None,
        help="extra experiment name for saving",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default='induction-exact',
        choices=['infini-gram', 'induction-exact', 'induction-fuzzy', 'induction-gram', 'llm'],
        help="type of model to evaluate",
    )                                                                                              
    parser.add_argument(
        "--fuzzy_checkpoint",
        type=str,
        default="gpt2",
        help="checkpoint for fuzzy matching",
    )
    parser.add_argument(
        "--tokenizer_checkpoint",
        type=str,
        default="gpt2",
        help="checkpoint for tokenizer",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        nargs='+',
        default=["v4_pileval_gpt2"],
        help="checkpoint for tokenizer",
    )

    # data args
    parser.add_argument(
        '--dataset',
        type=str,
        default='pile-val',
        help='dataset to use'
    )
    parser.add_argument(
        '--num_examples_test',
        type=int,
        default=100000,
        help='number of examples to test on'
    )
    parser.add_argument(
        "--infinigram_context_length",
        type=int,
        default=500,
        help="context length to generate data sample",
    )
    parser.add_argument(
        "--fuzzy_context_window_length",
        type=int,
        default=64,
        help="context length to generate data sample",
    )
    parser.add_argument(
        "--context_length",
        type=int,
        default=1024,
        help="context length to generate data sample",
    )
    parser.add_argument(
        "--data_stride",
        type=int,
        default=512,
        help="stride to generate data sample",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="batch size for evaluating"
    )

    return parser.parse_args()


if __name__ == '__main__':
    # hyperparams
    args = get_args()

    model_kwargs = dict(
        random_state=args.seed,
    )

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = 'cuda'

    # set up saving
    r = defaultdict(list)
    r.update(vars(args))
    if args.debug:
        args.save_dir = join(alm.config.BLOB_DIR, 'Exp/debug')
    _str_checkpoint = '.'.join([ckpt.split('/')[-1] for ckpt in args.checkpoint])
    _str_model = f'{args.model_type}_{args.tokenizer_checkpoint}'
    if args.model_type not in ['induction-fuzzy', 'induction-exact', 'llm']:
        _str_model += f'_{_str_checkpoint}'
    if args.model_type == 'llm':
        _str_model += f'_{args.checkpoint[0]}'
    if args.model_type in ['infini-gram', 'induction-gram']:
        _str_model += f'_infinicl{args.infinigram_context_length}'
    if args.model_type in ['induction-fuzzy', 'induction-gram']:
        _str_model += f'_fuzzycl{args.fuzzy_context_window_length}'
    elif args.model_type in ['induction-fuzzy', 'induction-gram']:
        if args.fuzzy_checkpoint.endswith('.pt'):
            _str_model += f'_fuzzy/{"_".join(args.fuzzy_checkpoint.split("/")[-5:-2])}'
        else:
            _str_model += f'_fuzzy-{args.fuzzy_checkpoint}'
    if args.exp_name is not None:
        _str_model += f'_{args.exp_name}'
    args.save_dir = join(args.save_dir, args.dataset, f'context{args.context_length}', _str_model)

    if os.path.exists(join(args.save_dir, 'prediction_total.csv')):
        data = pd.read_csv(f'{args.save_dir}/prediction.csv')
        if len(data['token_prob'].tolist()) >= args.num_examples_test:
            print("Evaluation is done!")
            exit(0)
        elif not args.resume:
            raise ValueError(f'{args.save_dir} already exists')
    elif not args.resume and os.path.exists(join(args.save_dir, 'prediction.csv')):
        raise ValueError(f'{args.save_dir} already exists')
    os.makedirs(args.save_dir, exist_ok=True)
    with open(join(args.save_dir, 'args.txt'), 'w') as f:
        for k, v in vars(args).items():
            f.write(f'{k}: {v}\n')

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(filename=join(args.save_dir, 'eval.log'), level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S%p')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)

    logger.info(f'Save Dir: {args.save_dir}')

    # set up data
    logger.info('Start loading data...')
    dset_test = alm.data.dsets.NextWordDataset(join(alm.config.DATA_DIR_ROOT, f'{args.dataset}_{args.tokenizer_checkpoint}'), context_length=args.context_length, stride=args.data_stride, split='test')

    if args.num_examples_test == -1:
        args.num_examples_test = len(dset_test)
    if len(dset_test) > args.num_examples_test:
        rng = np.random.default_rng(args.seed)
        example_nums = rng.choice(np.arange(len(dset_test)), size=args.num_examples_test, replace=False)
        sub_dset_test = Subset(dset_test, example_nums)
    else:
        sub_dset_test = dset_test
    test_loader = DataLoader(sub_dset_test, batch_size=args.batch_size, num_workers=4, drop_last=False)

    # 
    if args.tokenizer_checkpoint in LLM_MAP.keys():
        args.tokenizer_checkpoint = LLM_MAP[args.tokenizer_checkpoint]
    if args.model_type == 'llm' and args.checkpoint[0] in LLM_MAP.keys():
        args.checkpoint[0] = LLM_MAP[args.checkpoint[0]]
    if args.fuzzy_checkpoint in LLM_MAP.keys():
        args.fuzzy_checkpoint = LLM_MAP[args.fuzzy_checkpoint]

    # evaluate infini-gram perplexity on dset_test
    logger.info('Start building a model...')
    incontext_mode = None
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint, add_bos_token=False, add_eos_token=False, token=alm.config.TOKEN_HF)
    if args.model_type in ['infini-gram']:
        lm = InfiniGramMulti(
            load_to_ram=True,
            tokenizer=tokenizer,
            infinigram_checkpoints=[join(alm.config.INFINIGRAM_INDEX_PATH, ckpt) for ckpt in args.checkpoint],
            infinigram_context_length=args.infinigram_context_length,
            **model_kwargs
        )
    elif args.model_type in ['induction-exact', 'induction-fuzzy']:
        incontext_mode = args.model_type.split('-')[1]
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint, add_bos_token=False, add_eos_token=False, token=alm.config.TOKEN_HF)
        lm = InductionOnly(
            tokenizer=tokenizer,
            fuzzy_context_window_length=args.fuzzy_context_window_length,
            fuzzy_mm_name=args.fuzzy_checkpoint,
            **model_kwargs
        )
    elif args.model_type in ['induction-gram']:
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint, add_bos_token=False, add_eos_token=False, token=alm.config.TOKEN_HF)
        lm = InductionGram(
            tokenizer=tokenizer,
            fuzzy_mm_name=args.fuzzy_checkpoint if args.model_type in ['induction-gram', 'induction-gram-infsearch'] else None,
            fuzzy_context_window_length=args.fuzzy_context_window_length,
            infinigram_checkpoints=[join(alm.config.INFINIGRAM_INDEX_PATH, ckpt) for ckpt in args.checkpoint],
            infinigram_context_length=args.infinigram_context_length,
            **model_kwargs
        )
    elif args.model_type == 'llm':
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_checkpoint, token=alm.config.TOKEN_HF)
        lm = AutoModelForCausalLM.from_pretrained(args.checkpoint[0], token=alm.config.TOKEN_HF).eval().to(device)

    token_probs = []

    # should batch this to make it faster...
    recall_1 = []
    recall_5 = []
    recall_1_equal = []
    all_next_token_probs = []
    if args.model_type != 'llm':
        keys = ['query_count', 'token_prob', 'correct', 'suffix_len', 'sparse', 'recall_1', 'recall_5', 'recall_1_eq']
    else:
        keys = ['token_prob', 'correct', 'recall_1', 'recall_5', 'recall_1_eq']
    if args.model_type == 'induction-gram':
        keys += ['model']
    data_dict = {k: [] for k in keys}
    if args.resume and args.analyze and os.path.exists(f'{args.save_dir}/prediction.csv'):
        data = pd.read_csv(f'{args.save_dir}/prediction.csv')
        for k in keys:
            data_dict[k] = data[k].tolist()
        recall_1 = data_dict['recall_1'].copy()
        recall_5 = data_dict['recall_5'].copy()
        recall_1_equal = data_dict['recall_1_eq'].copy()
    
    n_eval_data = 0
    for batch_token_ids, batch_next_token_id in tqdm(test_loader, mininterval=5):
        if n_eval_data < len(data_dict['token_prob']):
            n_eval_data += len(batch_token_ids)
            continue
        if args.model_type == 'llm':
            with torch.no_grad():
                if 'llama' in args.checkpoint[0]:
                    batch_token_ids = torch.cat([torch.ones_like(batch_token_ids[:, :1]) * tokenizer.bos_token_id, batch_token_ids], dim=1)
                batch_token_ids = batch_token_ids.to(device)
                batch_next_token_logits = lm(batch_token_ids).logits[:, -1, :]
                batch_next_token_probs = torch.softmax(batch_next_token_logits, dim=-1).cpu()
        elif isinstance(lm, InductionGram):
            batch_next_token_probs, batch_others = lm.predict_prob(batch_token_ids,
                                                                   suffix_len_thres=8 if args.tokenizer_checkpoint == 'gpt2' else 9,
                                                                   split_size=2000 if args.tokenizer_checkpoint == 'gpt2' else 512,
                                                                   return_others=True)
        elif isinstance(lm, InfiniGramMulti):
            batch_next_token_probs, batch_others = lm.predict_prob(batch_token_ids, return_others=True)
        else:
            batch_next_token_probs, batch_others = lm.predict_prob(batch_token_ids, incontext_mode=incontext_mode, split_size=500, return_others=True)

        for i_batch in range(len(batch_token_ids)):
            token_ids, next_token_id = batch_token_ids[i_batch], batch_next_token_id[i_batch]
            next_token_probs = batch_next_token_probs[i_batch]
            _next_token_probs = next_token_probs[next_token_id].item()
            token_probs.append(_next_token_probs)
            recall_1.append(
                next_token_probs.argmax().item() == next_token_id.item())
            recall_5.append(
                next_token_id.item() in torch.Tensor(next_token_probs).topk(5).indices.tolist())
            recall_1_equal.append(
                (next_token_probs.max().item() == _next_token_probs) and (next_token_probs.min().item() != _next_token_probs))
        
            n_eval_data += 1

            if args.analyze:
                data_dict['recall_1'].append(next_token_probs.argmax().item() == next_token_id.item())
                data_dict['recall_5'].append(next_token_id.item() in torch.Tensor(next_token_probs).topk(5).indices.tolist())
                data_dict['recall_1_eq'].append((next_token_probs.max().item() == _next_token_probs) and (next_token_probs.min().item() != _next_token_probs))
                data_dict['token_prob'].append(_next_token_probs)
                data_dict['correct'].append('T' if recall_1_equal[-1] else 'F')
                if args.model_type != 'llm':
                    others = batch_others[i_batch]
                    data_dict['query_count'].append(others['prompt_cnt'])
                    data_dict['suffix_len'].append(others['suffix_len'])
                    data_dict['sparse'].append('T' if others['sparse'] else 'F')
                if args.model_type == 'induction-gram':
                    data_dict['model'].append(others['model'])
            
                if n_eval_data % 5000 == 0:
                    data_df = pd.DataFrame(data_dict)
                    data_df.to_csv(f'{args.save_dir}/prediction.csv')

                    logging.info("Analysis:")
                    logging.info(f"All: test equal recall@1: \t{data_df['correct'].tolist().count('T') / (data_df['correct'].tolist().count('T') + data_df['correct'].tolist().count('F'))}")
                    if args.debug:
                        exit(0)

    if args.analyze:
        data_df = pd.DataFrame(data_dict)
        data_df.to_csv(f'{args.save_dir}/prediction_total.csv')

    logging.info(f'test perfect recall@1: \t{np.mean(recall_1)}')
    logging.info(f'test perfect recall@5: \t{np.mean(recall_5)}')
    logging.info(f'test equal recall@1: \t{np.mean(recall_1_equal)}')

    if args.analyze:
        recall1_eq = data_df['correct'].tolist().count('T') / (data_df['correct'].tolist().count('T') + data_df['correct'].tolist().count('F'))
        logging.info("Analysis:")
        logging.info(f"All: test equal recall@1: \t{recall1_eq}")

        if args.model_type != 'llm':
            data_df = data_df[data_df['sparse'] == 'T']
            logging.info(f"# of sprase: {len(data_df)}")
            if len(data_df) > 0:
                logging.info(f"Sparse: test equal recall@1: {data_df['correct'].tolist().count('T') / (data_df['correct'].tolist().count('T') + data_df['correct'].tolist().count('F'))}")
