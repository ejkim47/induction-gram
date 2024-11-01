"""
This training script can be run both on a single gpu in debug mode,
and also in a larger training run with distributed data parallel (ddp).

To run on a single GPU, example:
$ python 01_train_fuzzymm.py --batch_size=32

To run with DDP on 4 gpus on 1 node, example:
$ torchrun --standalone --nproc_per_node=4 01_train_fuzzymm.py
"""

import os, glob
from os.path import join
import time
import math
import logging
from contextlib import nullcontext
from collections import defaultdict, deque
import argparse
import gc

import numpy as np
import torch
from torch.utils.data.dataloader import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group

from alm.data.dsets_fuzymm import SentenceDataset, SentenceDatasetfromSuffixArray
from alm.models.mini_gpt import GPTConfig, GPT
import alm.config
import wandb
import warnings

# Suppress FutureWarning messages
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

LLM_MAP = {'llama2-7B': 'meta-llama/Llama-2-7b-hf',
           'llama3-8B': 'meta-llama/Meta-Llama-3-8B',
           'llama2': 'meta-llama/Llama-2-7b-hf',
           'llama3': 'meta-llama/Meta-Llama-3-8B'}

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", type=bool, default=False, help="if debug")
    parser.add_argument("--seed", type=int, default=1, help="random seed")
    parser.add_argument(
        "--data_dir",
        type=str,
        default=alm.config.DATA_DIR_ROOT,
        help="directory for saving",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default=join(alm.config.SAVE_DIR_ROOT, "mechlm", "train_fuzzy_mm"),
        help="directory for saving",
    )
    parser.add_argument(
        "--exp_name",
        type=str,
        default='exp',
        help="directory for saving",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="gpt2",
        help="tokenizer",
    )
    parser.add_argument(
        "--fuzzy_context_window_length",
        type=int,
        default=32,
        help="context length of the model",
    )
    parser.add_argument(
        "--sampling_size",
        type=int,
        default=32,
        help="context length of the model",
    )
    parser.add_argument(
        "--sampling_size2",
        type=int,
        default=32,
        help="context length of the model",
    )
    parser.add_argument(
        '--init_from', type=str, default='gpt2', help='scratch or resume or gpt2 or llama2-7B', choices=['scratch', 'resume', 'gpt2', 'llama2-7B']
    )
    parser.add_argument(
        "--n_layer", type=int, default=4, help="# of layers"
    )
    parser.add_argument(
        "--n_head", type=int, default=12, help="# of heads"
    )
    parser.add_argument(
        "--n_embd", type=int, default=768, help="embedding dimension"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.1, help="temperature for cosine similarity"
    )
    parser.add_argument(
        "--n_out_embd", type=int, default=768, help="embedding dimension"
    )
    parser.add_argument(
        "--use_trained_pe", type=bool, default=False, help="if use trained positional embedding"
    )
    parser.add_argument(
        "--attention_block", type=str, default='RPETransformerBlock', choices=['TransformerBlock', 'RPETransformerBlock']
    )
    parser.add_argument(
        "--batch_size", type=int, default=2, help="batch size for training"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=0.0001, help="learning rate"
    )
    parser.add_argument(
        '--similarity_function', type=str, default='cosine', help='similarity function', choices=['cosine', 'jsd', 'l2']
    )
    parser.add_argument("--reduce_dim", type=bool, default=False, help="if reduce dimension to calculate jsd with teacher")
    parser.add_argument(
        '--num_epochs', type=int, default=2, help='number of epochs'
    )
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=16, help='gradient accumulation steps'
    )
    parser.add_argument(
        '--max_iters', type=int, default=20000,
    )
    parser.add_argument(
        '--teacher_llm', type=str, default='llama2-7B', help='teacher llm', choices=['gpt2', 'llama2-7B', 'llama3-8B']
    )
    parser.add_argument(
        '--lamb_ce', type=float, default=1., help='lambda for cross entropy'
    )
    parser.add_argument(
        '--lamb_kl', type=float, default=0., help='lambda for kl'
    )
    parser.add_argument(
        '--lamb_kl_reverse', type=float, default=1., help='lambda for kl reverse'
    )

    # data args
    parser.add_argument(
        '--dataset', type=str, default='openwebtext', help='dataset to use for training'
    )
    parser.add_argument(
        '--num_examples_test', type=int, default=2000, help='number of examples to test on'
    )
    parser.add_argument("--local-rank", type=int)

    return parser.parse_args()

if __name__ == '__main__':
    # hyperparams ######################
    args = get_args()
    
    # -----------------------------------------------------------------------------
    # default config values designed to train a gpt2 (124M) on OpenWebText
    # I/O
    eval_interval = 500
    log_interval = 10
    if args.debug:
        eval_interval = 10
        log_interval = 10
        args.num_examples_test = 100
    eval_only = True # if True, script exits right after the first eval
    # wandb logging
    wandb_log = False # disabled by default
    # data
    gradient_accumulation_steps = args.gradient_accumulation_steps # used to simulate larger batch sizesbat
    block_size = 1024
    batch_size = args.batch_size
    
    num_workers = 4
    # model
    dropout = 0.0 # for pretraining 0 is good, for finetuning try 0.1+
    bias = True # do we use bias inside LayerNorm and Linear layers?
    temperature = args.temperature
    # adamw optimizer
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.95
    grad_norm_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    # learning rate decay settings
    decay_lr = True # whether to decay the learning rate
    warmup_iters = 1000 # how many steps to warm up for
    # lr_decay_iters = 10000 # should be ~= max_iters per Chinchilla
    lr_decay_iters = args.max_iters
    min_lr = args.learning_rate * 0.1 # minimum learning rate, should be ~= learning_rate/10 per Chinchilla
    # system
    dtype = 'bfloat16' if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler

    # various inits, derived attributes, I/O setup
    backend = 'nccl' # 'nccl', 'gloo', etc.
    ddp = int(os.environ.get('RANK', -1)) != -1 # is this a ddp run?
    if ddp:
        init_process_group(backend=backend)
        ddp_rank = int(os.environ['RANK'])
        ddp_local_rank = int(os.environ['LOCAL_RANK'])
        ddp_world_size = int(os.environ['WORLD_SIZE'])
        device = f'cuda:{ddp_local_rank}'
        torch.cuda.set_device(device)
        master_process = ddp_rank == 0 # this process will do logging, checkpointing etc.
        seed_offset = ddp_rank # each process gets a different seed
        # world_size number of processes will be training simultaneously, so we can scale
        # down the desired gradient accumulation iterations per process proportionally
        assert gradient_accumulation_steps % ddp_world_size == 0
        gradient_accumulation_steps //= ddp_world_size
    else:
        # if not ddp, we are running on a single gpu, and one process
        master_process = True
        ddp_rank = seed_offset = 0
        ddp_world_size = 1
        device = 'cuda' # examples: 'cpu', 'cuda', 'cuda:0', 'cuda:1' etc., or try 'mps' on macbooks

    # -----------------------------------------------------------------------------
    config_keys = [k for k,v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
    # exec(open(os.path.join(alm.config.HDLM_EXP_DIR, 'configurator.py')).read()) # overrides from command line or config file
    config = {k: globals()[k] for k in config_keys} # will be useful for logging
    # -----------------------------------------------------------------------------

    # Set directory to save
    if args.sampling_size2 is None:
        args.sampling_size2 = args.sampling_size

    save_dir = join(args.save_dir, args.teacher_llm, args.exp_name)
    last_ckpt_index = -1
    existing_ckpt_list = glob.glob(join(save_dir, 'checkpoint/ckpt-last-*.pt'))
    if len(existing_ckpt_list) > 0:
        args.init_from = 'resume'
        last_ckpt_index = max([int(f.split('-')[-1].split('.')[0]) for f in existing_ckpt_list])
        ckpt_prelast_path = join(save_dir, f'checkpoint/ckpt-last-{last_ckpt_index:02d}.pt')
    ckpt_best_path = join(save_dir, f'checkpoint/ckpt-best-{last_ckpt_index+1:02d}.pt')
    ckpt_last_path = join(save_dir, f'checkpoint/ckpt-last-{last_ckpt_index+1:02d}.pt')

    if master_process:
        os.makedirs(join(save_dir, 'checkpoint'), exist_ok=True)

        with open(join(save_dir, 'args.txt'), 'w') as f:
            for k, v in vars(args).items():
                f.write(f'{k}: {v}\n')
            for k, v in config.items():
                f.write(f'{k}: {v}\n')
        
        if wandb_log:
            wandb.login(key=alm.config.WANDB_API_KEY)
            wandb_id, wandb_resume = None, False
            wandb_runfile = glob.glob(join(save_dir, 'wandb/run-*'))
            if args.init_from == 'resume' or len(wandb_runfile) > 0:
                wandb_id = wandb_runfile[0].split('-')[-1]
                wandb_resume = True
            config.update(vars(args))
            wandb_run = wandb.init(project='fuzzy-matching-model',
                                    name=join(args.teacher_llm, args.exp_name),
                                    config=config, dir=save_dir,
                                    resume=wandb_resume, id=wandb_id)
        
        # set up logging
        logger = logging.getLogger()
        logging.basicConfig(filename=join(save_dir, 'train.log'), level=logging.INFO, format='%(asctime)s %(message)s', datefmt='%m/%d/%Y %I:%M:%S%p')
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        logger.addHandler(stream_handler)

        logger.info(f'Save Dir: {save_dir}')

    torch.manual_seed(1337 + seed_offset)
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn
    device_type = 'cuda' if 'cuda' in device else 'cpu' # for later use in torch.autocast
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device_type == 'cpu' else torch.amp.autocast(device_type=device_type, dtype=ptdtype)

    if args.tokenizer == 'gpt2':
        tokenizer = AutoTokenizer.from_pretrained('gpt2', use_fast=False, add_bos_token=False, add_eos_token=False)
    elif args.tokenizer == 'llama2':
        tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf', token=alm.config.TOKEN_HF, use_fast=False, add_bos_token=False, add_eos_token=False) # The fast tokenizer seems unbearably slow ...
    eos_token_id = tokenizer.eos_token_id
    vocab_size = tokenizer.vocab_size

    # load the data
    if master_process:
        logger.info(f"Loading {args.dataset} data from {args.data_dir}")
    if args.dataset == 'openwebtext':
        data_train = SentenceDataset(data_dir=join(alm.config.DATA_DIR_ROOT, f'openwebtext_{args.tokenizer}'), max_n_tokens=args.fuzzy_context_window_length, min_n_tokens=args.fuzzy_context_window_length, split='train', pad_token_id=eos_token_id)
        data_val = SentenceDataset(data_dir=join(alm.config.DATA_DIR_ROOT, f'openwebtext_{args.tokenizer}'), max_n_tokens=args.fuzzy_context_window_length, min_n_tokens=args.fuzzy_context_window_length, split='val', pad_token_id=eos_token_id)
    elif args.dataset == 'pile':
        data_train = SentenceDatasetfromSuffixArray(data_dir=join(alm.config.DATA_DIR_ROOT, f'pile-uncopyrighted_{args.tokenizer}', 'train'),
                                                    max_n_tokens=args.fuzzy_context_window_length, min_n_tokens=args.fuzzy_context_window_length // 2, pad_token_id=eos_token_id)
        data_val = SentenceDatasetfromSuffixArray(data_dir=join(alm.config.DATA_DIR_ROOT, f'pile-uncopyrighted_{args.tokenizer}', 'val'),
                                                    max_n_tokens=args.fuzzy_context_window_length, min_n_tokens=args.fuzzy_context_window_length // 2, pad_token_id=eos_token_id)
    else:
        raise ValueError(f"Argument dataset should be one of ['openwebtext', 'pile'], but got {args.dataset}")

    # init these up here, can override if init_from='resume' (i.e. from a checkpoint)
    iter_num, iter_num_restart = 0, 0
    best_val_loss = 1e9

    # model init
    model_kwargs = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=block_size, n_out_embd=args.n_out_embd,
                        bias=bias, vocab_size=vocab_size, dropout=dropout,
                        use_trained_pe=args.use_trained_pe,
                        similarity_function=args.similarity_function,
                        attention_block=args.attention_block, max_relative_position=args.fuzzy_context_window_length) # start with model_kwargs from command line
    if args.init_from == 'scratch':
        # init a new model from scratch
        if master_process:
            logger.info("Initializing a new model from scratch")
        gptconf = GPTConfig(**model_kwargs)
        model = GPT(gptconf)
    elif args.init_from.startswith('gpt2'):
        if master_process:
            logger.info(f"Initializing from OpenAI GPT-2 weights")
        # initialize from OpenAI GPT-2 weights
        override_args = dict(n_layer=args.n_layer, n_head=args.n_head, block_size=block_size, n_out_embd=args.n_out_embd,
                             bias=bias, dropout=dropout, use_trained_pe=args.use_trained_pe, 
                             similarity_function=args.similarity_function, attention_block=args.attention_block, max_relative_position=args.fuzzy_context_window_length)
        model = GPT.from_pretrained(args.init_from, override_args)
        # read off the created config params, so we can store them into checkpoint correctly
        for k in ['vocab_size', 'n_embd']:
            model_kwargs[k] = getattr(model.config, k)
    elif args.init_from == 'llama2-7B':
        if master_process:
            logger.info("Initializing wte from LLaMA-2-7B")
        initialized_llm = AutoModelForCausalLM.from_pretrained('meta-llama/Llama-2-7b-hf', token=alm.config.TOKEN_HF, device_map='cpu')
        assert model_kwargs['vocab_size'] == initialized_llm.model.embed_tokens.weight.shape[0]
        assert model_kwargs['n_embd'] == initialized_llm.model.embed_tokens.weight.shape[1]
        gptconf = GPTConfig(**model_kwargs)
        model = GPT(gptconf)
        model.transformer.wte.weight.data = initialized_llm.model.embed_tokens.weight.data
        del initialized_llm
    elif args.init_from == 'resume':
        if master_process:
            logger.info(f"Resuming training from {save_dir}")
        # resume training from a checkpoint.
        checkpoint = torch.load(ckpt_prelast_path, map_location=device)
        checkpoint_model_kwargs = checkpoint['model_kwargs']
        # force these config attributes to be equal otherwise we can't even resume training
        # the rest of the attributes (e.g. dropout) can stay as desired from command line
        for k, v in checkpoint_model_kwargs.items():
            model_kwargs[k] = v
        # create the model
        gptconf = GPTConfig(**model_kwargs)
        model = GPT(gptconf)
        state_dict = checkpoint['model']
        model.load_state_dict(state_dict)
        iter_num_restart = checkpoint['iter_num']
        best_val_loss = checkpoint['best_val_loss']
        args.init_from = True

    # crop down the model block size if desired, using model surgery
    if block_size < model.config.block_size:
        model.crop_block_size(block_size)
        model_kwargs['block_size'] = block_size # so that the checkpoint will have the right value
    
    # wrap model into DDP container
    model.to(device)
    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    load_in_8bit = False
    if args.teacher_llm in LLM_MAP.keys():
        args.teacher_llm = LLM_MAP[args.teacher_llm]
        load_in_8bit = True
    tokenizer_teacher = AutoTokenizer.from_pretrained(args.teacher_llm, token=alm.config.TOKEN_HF)
    tokenizer_teacher.pad_token = tokenizer_teacher.eos_token
    teacher_llm = AutoModelForCausalLM.from_pretrained(args.teacher_llm, token=alm.config.TOKEN_HF, load_in_8bit=load_in_8bit,
                                                       device_map=device)
    if args.teacher_llm == 'gpt2':
        teacher_llm = teacher_llm.eval().to(device)

    # optimizer
    optimizer = model.configure_optimizers(learning_rate=args.learning_rate, weight_decay=config['weight_decay'],
                                           betas=(config['beta1'], config['beta2']), device_type=config['device'])
    if args.init_from == 'resume':
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint

    if ddp:
        model = DDP(model, device_ids=[ddp_local_rank])

    # learning rate decay scheduler (cosine with warmup)
    def get_lr(it):
        # 1) linear warmup for warmup_iters steps
        if it < warmup_iters:
            return args.learning_rate * it / warmup_iters
        # 2) if it > lr_decay_iters, return min learning rate
        if it > lr_decay_iters:
            return min_lr
        # 3) in between, use cosine decay down to min learning rate
        decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
        assert 0 <= decay_ratio <= 1
        coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
        return min_lr + coeff * (args.learning_rate - min_lr)
    
    # training loop
    t0 = time.time()
    
    # set up saving
    r = defaultdict(list)
    r.update(vars(args))

    # setup the dataloader
    train_loader = DataLoader(
        data_train,
        sampler=torch.utils.data.RandomSampler(data_train, replacement=True, num_samples=int(1e10)),
        shuffle=False,
        pin_memory=True,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )
    val_loader = DataLoader(
        data_val,
        sampler=torch.utils.data.RandomSampler(data_val, replacement=True, num_samples=args.num_examples_test),
        shuffle=False,
        pin_memory=True,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
    )

    def forward_teacher_llm(x, attention_mask, indices1, indices2):
        with torch.no_grad():
            if tokenizer_teacher.name_or_path != tokenizer.name_or_path:
                indices_v, indices_i = torch.cat([indices1, indices2]).sort()
                mask1 = indices_i < len(indices1)
                
                _new_x, new_indices_mask = [], []
                inputs_teacher = tokenizer_teacher([tokenizer.decode(x[i, :j+1].cpu()) for i, j in zip(indices_v // args.fuzzy_context_window_length, indices_v % args.fuzzy_context_window_length)], return_tensors="pt", padding=True)
                rep = (inputs_teacher['input_ids'][:-1] == inputs_teacher['input_ids'][1:])
                rep_clone = rep.clone()
                rep[inputs_teacher['attention_mask'][:-1] == 0] = True
                _mask = (~rep.all(1)) | rep_clone.all(1)
                idx_list = torch.arange(len(rep))[_mask]
                idx_list = torch.cat([torch.ones_like(idx_list[:1]) * (-1), idx_list, torch.ones_like(idx_list[-1:]) * len(rep)])
                batch_idx = torch.cat([(indices_v // args.fuzzy_context_window_length)[:-1][_mask], (indices_v // args.fuzzy_context_window_length)[-1:]])
                attention_pos = inputs_teacher['attention_mask'] - torch.cat([inputs_teacher['attention_mask'][:, 1:], torch.zeros_like(inputs_teacher['attention_mask'][:, -1:])], dim=1)
                for s_idx, e_idx, b_idx in zip(idx_list[:-1], idx_list[1:], batch_idx.cpu()):
                    _new_x.append(inputs_teacher['input_ids'][e_idx])
                    new_indices_mask.append(attention_pos[s_idx+1:e_idx+1].sum(0) * (b_idx + 1))
                new_indices_mask = torch.stack(new_indices_mask).to(indices_v.device)
                _new_x = torch.stack(_new_x, dim=0)

                logits = teacher_llm(_new_x.to(x.device), return_dict=True).logits
                for re_idx in torch.isnan(logits[..., 0]).any(1).nonzero().view(-1):
                    logits[re_idx:re_idx+1] = teacher_llm(_new_x[re_idx:re_idx+1].to(x.device), return_dict=True).logits
                logits_reshape = logits[new_indices_mask.cpu() > 0]
                logits_reshape1 = logits_reshape[mask1]
                logits_reshape2 = logits_reshape[~mask1]

            else:
                new_x, new_attention_mask = x, attention_mask
                logits = teacher_llm(new_x, attention_mask=new_attention_mask, return_dict=True).logits
                for re_idx in torch.isnan(logits[..., 0]).any(1).nonzero().view(-1):
                    logits[re_idx:re_idx+1] = teacher_llm(_new_x[re_idx:re_idx+1].to(x.device), return_dict=True).logits
                logits = logits.view(-1, logits.size(-1))
                logits_reshape1 = logits[indices1]
                logits_reshape2 = logits[indices2]

            if args.reduce_dim:
                _values, _indices = logits_reshape2.softmax(dim=-1).sort(descending=True, dim=1)
                selected2 = _values.cumsum(dim=1) < 0.9
                selected2 = torch.cat([selected2[..., -1:], selected2[..., :-1]], dim=-1)
                selected2[:, 0] = True
                selected2 = _indices[selected2].unique()

                _values, _indices = logits_reshape1.softmax(dim=-1).sort(descending=True, dim=1)
                selected1 = _values.cumsum(dim=1) < 0.9
                selected1 = torch.cat([selected1[..., -1:], selected1[..., :-1]], dim=-1)
                selected1[:, 0] = True
                selected1 = _indices[selected1].unique()
            else:
                selected1 = selected2 = torch.arange(logits_reshape1.size(-1))

            logit_log_probs_reshape1 = logits_reshape1.log_softmax(dim=-1)
            logit_log_probs_reshape2 = logits_reshape2.log_softmax(dim=-1)

            all_distance = []
            torch.cuda.empty_cache()
            for _logit_log_probs_reshape1 in logit_log_probs_reshape1.split(128):
                distance = (logit_log_probs_reshape2[None, :, selected2].exp() * (logit_log_probs_reshape2[None, :, selected2] - _logit_log_probs_reshape1[:, None, selected2])).sum(dim=-1)
                torch.cuda.empty_cache()
                distance += (_logit_log_probs_reshape1[:, None, selected1].exp() * (_logit_log_probs_reshape1[:, None, selected1] - logit_log_probs_reshape2[None, :, selected1])).sum(dim=-1)
                torch.cuda.empty_cache()
                distance = distance / 2
                all_distance.append(distance)
            
            all_distance = torch.cat(all_distance, dim=0)
        
        indices1 = indices1[~(all_distance != all_distance).all(1)]
        indices2 = indices2[~(all_distance != all_distance).all(0)]
        all_distance = all_distance[~(all_distance != all_distance).all(1)][:,  ~(all_distance != all_distance).all(0)]

        return all_distance, indices1, indices2

    raw_model = model.module if ddp else model # unwrap DDP container if needed
    if master_process:
        logger.info("Starting training...")
    model.train()
    iter_num, local_num = 0, 0
    iter_time = time.time()
    loss_hist = deque(maxlen=log_interval * gradient_accumulation_steps)

    data_iter = iter(train_loader)
    while True:
        # determine and set the learning rate for this iteration
        lr = get_lr(iter_num) if decay_lr else args.learning_rate
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        if iter_num < iter_num_restart:
            iter_num += 1
            local_num += gradient_accumulation_steps
            continue

        for micro_step in range(gradient_accumulation_steps):
            if ddp:
                # in DDP training we only need to sync gradients at the last micro step.
                # the official way to do this is with model.no_sync() context manager, but
                # I really dislike that this bloats the code and forces us to repeat code
                # looking at the source of that context manager, it just toggles this variable
                model.require_backward_grad_sync = (micro_step == gradient_accumulation_steps - 1)
            # fetch the next batch (x, y) and re-init iterator if needed
            with ctx:
                try:
                    _, batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_loader)
                    _, batch = next(data_iter)
                
                x = batch.to(device)

                # Forward the teacher model
                if tokenizer_teacher.name_or_path != tokenizer.name_or_path:
                    max_len = tokenizer_teacher([tokenizer.decode(_x) for _x in x.cpu()], return_tensors="pt", padding=True)['attention_mask'].sum(1)
                    x = x[max_len < 4 * args.fuzzy_context_window_length]

                attention_mask = (x != eos_token_id).float()
                assert x.max() < vocab_size, f"out of bounds: {x.max()}"

                indices = torch.arange(x.shape[0] * x.shape[1], device=attention_mask.device).view_as(x)[attention_mask > 0]
                indices_perm = indices[torch.randperm(len(indices))]
                if len(indices_perm) < args.sampling_size + args.sampling_size2:
                    indices1, indices2 = indices_perm[:-args.sampling_size2], indices_perm[-args.sampling_size2:]
                else:
                    indices1, indices2 = indices_perm[:args.sampling_size], indices_perm[args.sampling_size:args.sampling_size + args.sampling_size2]

                indices1 = indices1.sort().values
                indices2 = indices2.sort().values
                t_distance, indices1, indices2 = forward_teacher_llm(x, attention_mask, indices1, indices2)
                
                # There are some cases that the logit score is NaN, so we need to delete it
                indices1 = indices1[~(t_distance != t_distance).all(1)]
                indices2 = indices2[~(t_distance != t_distance).all(0)]
                t_distance = t_distance[~(t_distance != t_distance).all(1)][:,  ~(t_distance != t_distance).all(0)]
                
                # forward the student model
                logits, distance, loss = model(x.to(device), indices=indices1, indices2=indices2, targets=t_distance.argmin(dim=1), distance_targets=t_distance.detach(), temperature=temperature,
                                            lamb_ce=args.lamb_ce, lamb_kl=args.lamb_kl, lamb_kl_reverse=args.lamb_kl_reverse)
                
                loss = loss / gradient_accumulation_steps # scale the loss to account for gradient accumulation
            scaler.scale(loss).backward()
            local_num += 1
        # clip the gradient
        if config['grad_norm_clip'] != 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config['grad_norm_clip'])
        # step the optimizer and scaler if training in fp16
        scaler.step(optimizer)
        scaler.update()
        # flush the gradients as soon as we can, no need for this memory anymore
        optimizer.zero_grad(set_to_none=True)
        iter_num += 1
        tnow = time.time()

        loss_hist.append(loss.item() * gradient_accumulation_steps)

        # Log the training metrics
        if iter_num % log_interval == 0 and master_process:
            loss = np.array(loss_hist).mean()
            with torch.no_grad():
                pred_rank = distance.argsort(dim=-1)
                labels_rank = t_distance.argsort(dim=-1)
                recall1 = (pred_rank[:, 0] == labels_rank[:, 0]).float().mean().item()
                recall5 = (pred_rank[:, :5] == labels_rank[:, :1]).float().sum(1).mean().item()
            logging.info(f"iter {iter_num}, train loss={loss:.4f}, train recall1={recall1:.4f}, train recall5={recall5:.4f}, time={tnow - iter_time:.2f}s")
            
            lr_dict = {f'training/lr_{g_idx}': p_group['lr'] for g_idx, p_group in enumerate(optimizer.param_groups)}
            if wandb_log:
                wandb.log({'loss/train': loss, 'metrics/train_recall1': recall1, 'metrics/train_recall5': recall5}, step=iter_num)
                wandb.log(lr_dict, step=iter_num)
        
        gc.collect()
        with torch.no_grad():
            torch.cuda.empty_cache()
        # Log the evaluation metrics
        if iter_num % eval_interval == 0 and master_process:
            model.eval()
            data_iter = iter(val_loader)
            loss_hist_val = []
            target_rank_all, pred_rank_all, mrr_rank_all = [], [], []
            with torch.no_grad():
                iter_val = 0
                while True:
                    try:
                        _, batch = next(data_iter)
                    except StopIteration:
                        data_iter = iter(val_loader)
                        _, batch = next(data_iter)
                    
                    x = batch.to(device)

                    # Forward the teacher model
                    if args.teacher_llm != args.tokenizer:
                        max_len = tokenizer_teacher([tokenizer.decode(_x) for _x in x.cpu()], return_tensors="pt", padding=True)['attention_mask'].sum(1)
                        x = x[max_len < 4 * args.fuzzy_context_window_length]

                    attention_mask = (x != eos_token_id).float()
                    assert x.max() < vocab_size, f"out of bounds: {x.max()}"

                    indices = torch.arange(x.shape[0] * x.shape[1], device=attention_mask.device).view_as(x)[attention_mask > 0]
                    indices_perm = indices[torch.randperm(len(indices))]
                    if len(indices_perm) < args.sampling_size + args.sampling_size2:
                        indices1, indices2 = indices_perm[:-args.sampling_size2], indices_perm[-args.sampling_size2:]
                    else:
                        indices1, indices2 = indices_perm[:args.sampling_size], indices_perm[args.sampling_size:args.sampling_size + args.sampling_size2]

                    indices1 = indices1.sort().values
                    indices2 = indices2.sort().values
                    t_distance, indices1, indices2 = forward_teacher_llm(x, attention_mask, indices1, indices2)
                    
                    indices1 = indices1.sort().values
                    indices2 = indices2.sort().values
                    labels_rank = t_distance.argsort(dim=-1)

                    target_rank_all.append(labels_rank[:, :5].detach().cpu())
                    # forward the model
                    logits, distance, loss = model(x.to(device), indices=indices1, indices2=indices2, targets=t_distance.argmin(dim=1).detach(), distance_targets=t_distance.detach(), temperature=temperature,
                                                   lamb_ce=args.lamb_ce, lamb_kl=args.lamb_kl, lamb_kl_reverse=args.lamb_kl_reverse)
                    pred_rank = distance.argsort(dim=-1).detach().cpu()
                    pred_rank_all.append(pred_rank[:, :5])
                    mrr_rank_all.append(1 / ((pred_rank == labels_rank[:, :1].detach().cpu()).int().argmax(1) + 1))
                    loss_hist_val.append(loss.item())

                    iter_val += 1
                    if len(loss_hist_val) > args.num_examples_test:
                        break
            
            loss = np.array(loss_hist_val).mean()
            pred_rank_all = torch.cat(pred_rank_all, dim=0)
            target_rank_all = torch.cat(target_rank_all, dim=0)
            recall1 = (pred_rank_all[:, 0] == target_rank_all[:, 0]).float().mean()
            recall5 = (pred_rank_all[:, :5] == target_rank_all[:, :1]).float().sum(1).mean()
            mrr = torch.cat(mrr_rank_all).mean()
            logging.info(f"iter {iter_num}, val loss={loss:.4f}, val recall1={recall1:.4f}, val recall5={recall5:.4f}, val mrr={mrr:.4f}, time={time.time() - iter_time:.2f}s")
            if wandb_log:
                wandb.log({'loss/val': loss, 'metrics/val_recall1': recall1, 'metrics/val_recall5': recall5, 'metrics/val_mrr': mrr}, step=iter_num)
            if best_val_loss > loss:
                best_val_loss = loss
                checkpoint = {
                    'model': raw_model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'model_kwargs': model_kwargs,
                    'iter_num': iter_num,
                    'best_val_loss': best_val_loss,
                    'config': config,
                }
                logging.info(f"Saving checkpoint to {ckpt_best_path}")
                torch.save(checkpoint, ckpt_best_path)
            checkpoint = {
                'model': raw_model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'model_kwargs': model_kwargs,
                'iter_num': iter_num,
                'best_val_loss': best_val_loss,
                'config': config,
            }
            logging.info(f"Saving checkpoint to {ckpt_last_path}")
            torch.save(checkpoint, ckpt_last_path)
            
            model.train()

        # termination conditions
        if args.max_iters is not None and iter_num >= args.max_iters:
            break
    
    if ddp:
        destroy_process_group()