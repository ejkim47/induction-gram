import argparse
import multiprocessing as mp
import numpy as np
import os
import resource
import sys
from tqdm import tqdm
from datasets import load_dataset
import random
import logging
from infini_gram.indexing import build_sa

import alm.config

HACK = 100000

tokenizer = None

def tok(line):
    global tokenizer
    tok_text = tokenizer.encode(line)
    byte_arr = np.array(tok_text, dtype=np.uint16).view(np.uint8).tobytes()
    return byte_arr, len(tok_text)

def tokenize(args, logger):

    ds_paths = [os.path.join(args.save_dir, f'tokenized.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    od_paths = [os.path.join(args.save_dir, f'offset.{i}') for i in range(args.worker_id, args.shards, args.workers)]
    if all([os.path.exists(ds_path) for ds_path in ds_paths]) \
        and all([os.path.exists(od_path) for od_path in od_paths]):
        logger.info('Step 1 (tokenize): Skipped. All tokenized files already exist.')
        return

    logger.info('Step 1 (tokenize): Starting ...')

    import transformers
    transformers.utils.logging.set_verbosity(40) # suppress warnings
    global tokenizer
    if args.tokenizer == 'gpt2':
        tokenizer = transformers.AutoTokenizer.from_pretrained('gpt2', use_fast=False, add_bos_token=False, add_eos_token=False)
    elif args.tokenizer == 'llama2':
        tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=alm.config.TOKEN_HF, use_fast=False, add_bos_token=False, add_eos_token=False) # The fast tokenizer seems unbearably slow ...
    elif args.tokenizer == 'olmo':
        tokenizer = transformers.AutoTokenizer.from_pretrained("allenai/OLMo-7B", add_bos_token=False, add_eos_token=False)
        # # The following is a faster version, but the result is a bit different
        # from dolma.tokenizer import Tokenizer
        # tokenizer = Tokenizer.from_pretrained('allenai/gpt-neox-olmo-dolma-v1_5', bos_token_id=None, eos_token_id=None, pad_token_id=1, segment_before_tokenization=True)
    else:
        raise ValueError(f'Unknown tokenizer: {args.tokenizer}')

    ds_fouts = [open(ds_path, 'wb') for ds_path in ds_paths]
    od_fouts = [open(od_path, 'wb') for od_path in od_paths]
    with mp.get_context('fork').Pool(args.cpus) as p:
        ods = [0 for _ in od_fouts]

        total_n_tokens = 0
        if args.dataset_name == 'fineweb':
            data = load_dataset("HuggingFaceFW/fineweb", name="sample-10BT", split="train", streaming=True)
            import torch
            dl = torch.utils.data.DataLoader(data, num_workers=8, batch_size=args.batch_size)
            for d in dl:
                batch_lines = d['text']
                results = p.map(tok, batch_lines)
                for i, (byte_arr, n_token) in enumerate(results):
                    content = args.doc_sep + byte_arr
                    j = i % (args.shards // args.workers)
                    ds_fouts[j].write(content)
                    od_fouts[j].write(np.array([ods[j]], dtype=np.uint64).view(np.uint8).tobytes())
                    ods[j] += len(content)
                    total_n_tokens += n_token
            del data
        elif args.dataset_name == 'babylm':
            data_paths = [os.path.join(args.data_dir, f) for f in os.listdir(args.data_dir) if f.endswith('.dev' if args.split == 'val' else '.test')]
            logger.info("Found data paths:")
            for data_path in data_paths:
                logger.info('\t' + data_path)
            for data_path in tqdm(data_paths):
                lines = [open(data_path).read()]
                for offset in tqdm(range(0, len(lines), args.workers*args.batch_size), total=len(range(0, len(lines), args.workers*args.batch_size)), mininterval=1):
                    batch_lines = lines[(offset+args.worker_id):(offset+args.workers*args.batch_size):args.workers]
                    results = p.map(tok, batch_lines)
                    for i, (byte_arr, n_token) in enumerate(results):
                        content = args.doc_sep + byte_arr
                        j = i % (args.shards // args.workers)
                        ds_fouts[j].write(content)
                        od_fouts[j].write(np.array([ods[j]], dtype=np.uint64).view(np.uint8).tobytes())
                        ods[j] += len(content)
                        total_n_tokens += n_token
                del lines
        elif args.dataset_name == 'pile-uncopyrighted':
            data = load_dataset("monology/pile-uncopyrighted", split='test')
            lines = data['text']
            for offset in tqdm(range(0, len(lines), args.workers*args.batch_size), total=len(range(0, len(lines), args.workers*args.batch_size)), mininterval=1):
                batch_lines = lines[(offset+args.worker_id):(offset+args.workers*args.batch_size):args.workers]
                results = p.map(tok, batch_lines)
                for i, (byte_arr, n_token) in enumerate(results):
                    content = args.doc_sep + byte_arr
                    j = i % (args.shards // args.workers)
                    ds_fouts[j].write(content)
                    od_fouts[j].write(np.array([ods[j]], dtype=np.uint64).view(np.uint8).tobytes())
                    ods[j] += len(content)
                    total_n_tokens += n_token
            del data
            del lines
    if total_n_tokens//1e9 > 1:
        logger.info(f'Loading tokenized documents (# of tokens: {int(total_n_tokens//1e9)}.{int((total_n_tokens%1e9)//1e6)}B)')
    else:
        logger.info(f'Loading tokenized documents (# of tokens: {int(total_n_tokens//1e6)}M)')

    for ds_fout in ds_fouts:
        ds_fout.close()
    for od_fout in od_fouts:
        od_fout.close()

        
def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1, help='Random seed.')
    parser.add_argument('--split', type=str, default='test', help='Dataset split.')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing the raw text corpus. Must be absolute path.')
    parser.add_argument('--dataset_name', type=str, default='babylm', choices=['babylm', 'fineweb', 'pile-uncopyrighted'], help='Name of the dataset.')
    parser.add_argument('--temp_dir', type=str, default=None, help='Directory where temporary indexing files are stored. Must be absolute path.')
    parser.add_argument('--save_dir', type=str, default=alm.config.DATA_DIR_ROOT, help='Directory where the final index files are stored. Must be absolute path.')
    parser.add_argument('--tokenizer', type=str, required=True, choices=['gpt2', 'llama2', 'olmo'])
    parser.add_argument('--doc_sep', type=bytes, default=b'\xff\xff')
    parser.add_argument('--batch_size', type=int, default=65536, help='Batch size for tokenization.')
    parser.add_argument('--cpus', type=int, default=mp.cpu_count(), help='Number of CPU cores available to the program.')
    parser.add_argument('--mem', type=int, required=True, help='Amount of memory in GiB available to the program.')
    parser.add_argument('--shards', type=int, default=1, help='Number of shards to split the index into.')
    parser.add_argument('--workers', type=int, default=1, help='Total number of workers. Must be a divisor of shards.')
    parser.add_argument('--worker_id', type=int, default=0, help='The worker ID of this process. Must be in range [0, workers).')
    parser.add_argument('--add_metadata', default=False, action='store_true', help='Whether to store document metadata in the index.')
    parser.add_argument('--ulimit', type=int, default=1048576, help='Maximum number of open files allowed.')
    args = parser.parse_args()

    args.save_dir = os.path.join(args.save_dir, f'{args.dataset_name}_{args.tokenizer}', args.split)

    if args.temp_dir is None:
        args.temp_dir = args.save_dir
    args.data_dir = args.data_dir.rstrip('/')
    args.temp_dir = args.temp_dir.rstrip('/')
    args.save_dir = args.save_dir.rstrip('/')

    assert args.batch_size > 0
    assert args.cpus > 0
    assert args.shards > 0
    assert args.workers > 0
    assert 0 <= args.worker_id < args.workers
    assert args.shards % args.workers == 0

    os.makedirs(args.temp_dir, exist_ok=True)
    os.makedirs(args.save_dir, exist_ok=True)

    # set up logging
    logger = logging.getLogger()
    logging.basicConfig(filename=os.path.join(args.save_dir, f'log_{args.worker_id}.log'), level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S%p')
    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(logging.INFO)
    logger.addHandler(stream_handler)


    assert sys.byteorder == 'little'
    resource.setrlimit(resource.RLIMIT_NOFILE, (args.ulimit, args.ulimit))

    # set seed
    np.random.seed(args.seed)
    random.seed(args.seed)

    tokenize(args, logger)
    # build_sa(args)

if __name__ == '__main__':
    main()
