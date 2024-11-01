import math
import inspect
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

def get_emb(sin_inp):
    """
    Gets a base embedding for one dimension with sin and cos intertwined
    """
    emb = torch.stack((sin_inp.sin(), sin_inp.cos()), dim=-1)
    return torch.flatten(emb, -2, -1)

class PositionalEncoding1D(nn.Module):
    def __init__(self, channels):
        """
        :param channels: The last dimension of the tensor you want to apply pos emb to.
        """
        super(PositionalEncoding1D, self).__init__()
        self.org_channels = channels
        channels = int(np.ceil(channels / 2) * 2)
        self.channels = channels
        inv_freq = 1.0 / (10000 ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        self.register_buffer("cached_penc", None, persistent=False)

    def forward(self, tensor):
        """
        :param tensor: A 3d tensor of size (batch_size, x, ch)
        :return: Positional Encoding Matrix of size (batch_size, x, ch)
        """
        if len(tensor.shape) != 3:
            raise RuntimeError("The input tensor has to be 3d!")

        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        inv_freq = 1.0 / (10000 ** (torch.arange(0, self.channels, 2).float() / self.channels))
        batch_size, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device, dtype=inv_freq.dtype)
        sin_inp_x = torch.einsum("i,j->ij", pos_x, inv_freq.to(tensor.device))
        emb_x = get_emb(sin_inp_x)
        emb = torch.zeros((x, self.channels), device=tensor.device, dtype=tensor.dtype)
        emb[:, : self.channels] = emb_x

        self.cached_penc = emb[None, :, :orig_ch].repeat(batch_size, 1, 1)
        return self.cached_penc

class RelativePosition(nn.Module):

    def __init__(self, num_units, max_relative_position):
        super().__init__()
        self.num_units = num_units
        self.max_relative_position = max_relative_position
        self.embeddings_table = nn.Parameter(torch.Tensor(max_relative_position * 2 + 1, num_units))
        nn.init.xavier_uniform_(self.embeddings_table)

    def forward(self, length_q, length_k):
        range_vec_q = torch.arange(length_q)
        range_vec_k = torch.arange(length_k)
        distance_mat = range_vec_k[None, :] - range_vec_q[:, None]
        distance_mat_clipped = torch.clamp(distance_mat, -self.max_relative_position, self.max_relative_position)
        final_mat = distance_mat_clipped + self.max_relative_position
        final_mat = torch.LongTensor(final_mat).cuda()
        embeddings = self.embeddings_table[final_mat].cuda()

        return embeddings

class LayerNorm(nn.Module):
    """ LayerNorm but with an optional bias. PyTorch doesn't support simply bias=False """

    def __init__(self, ndim, bias):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, input):
        return F.layer_norm(input, self.weight.shape, self.weight, self.bias, 1e-5)

class CausalSelfAttention(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.n_embd % config.n_head == 0
        # key, query, value projections for all heads, but in a batch
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        # regularization
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        # flash attention make GPU go brrrrr but support is only in PyTorch >= 2.0
        self.flash = hasattr(torch.nn.functional, 'scaled_dot_product_attention')
        if not self.flash:
            print("WARNING: using slow attention. Flash Attention requires PyTorch >= 2.0")
            # causal mask to ensure that attention is only applied to the left in the input sequence
            self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                        .view(1, 1, config.block_size, config.block_size))

    def forward(self, x):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)

        # causal self-attention; Self-attend: (B, nh, T, hs) x (B, nh, hs, T) -> (B, nh, T, T)
        if self.flash:
            # efficient attention using Flash Attention CUDA kernels
            y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=None, dropout_p=self.dropout if self.training else 0, is_causal=True)
        else:
            # manual implementation of attention
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        y = y.transpose(1, 2).contiguous().view(B, T, C) # re-assemble all head outputs side by side

        # output projection
        y = self.resid_dropout(self.c_proj(y))
        return y

class RPESelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()

        assert config.n_embd % config.n_head == 0

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout
        self.head_dim = config.n_embd // config.n_head
        self.max_relative_position = config.max_relative_position

        self.relative_position_k = RelativePosition(self.head_dim, self.max_relative_position)
        self.relative_position_v = RelativePosition(self.head_dim, self.max_relative_position)

        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)
        
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # causal mask to ensure that attention is only applied to the left in the input sequence
        self.register_buffer("bias", torch.tril(torch.ones(config.block_size, config.block_size))
                                    .view(1, 1, config.block_size, config.block_size))
        
    def forward(self, x, past_k=None, past_v=None, return_kv=False):
        B, T, C = x.size() # batch size, sequence length, embedding dimensionality (n_embd)

        q, k, v  = self.c_attn(x).split(self.n_embd, dim=2)
        r_k1 = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        r_q1 = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        r_v1 = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, nh, T, hs)
        Tq, Tkv = T, T

        if past_k is not None and past_v is not None:
            assert (past_k.shape == past_v.shape).all()
            Tkv = Tkv + past_k.shape[2]
            r_k1 = torch.cat([past_k, r_k1], dim=2)
            r_v1 = torch.cat([past_v, r_v1], dim=2)

        attn1 = torch.matmul(r_q1, r_k1.transpose(-2, -1)) 

        r_q2 = q.transpose(0, 1).contiguous().view(Tq, B*self.n_head, C // self.n_head)
        r_k2 = self.relative_position_k(T, T)
        attn2 = torch.matmul(r_q2, r_k2.transpose(1, 2)).transpose(0, 1)
        attn2 = attn2.contiguous().view(B, self.n_head, T, T)
        att = (attn1 + attn2) * (1.0 / math.sqrt(r_k1.size(-1)))

        att = att.masked_fill(self.bias[:,:,:T,:T] == 0, float('-inf'))
        mask = torch.tril(torch.ones_like(self.bias[:,:,:T,:T]), diagonal=-self.max_relative_position).view(1, 1, T, T)
        att = att.masked_fill(mask == 1, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.attn_dropout(att)

        weight1 = att @ r_v1 # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)
        r_v2 = self.relative_position_v(T, T)
        weight2 = att.permute(2, 0, 1, 3).contiguous().view(Tq, B * self.n_head, Tkv)
        weight2 = weight2 @ r_v2
        weight2 = weight2.transpose(0, 1).contiguous().view(B, self.n_head, Tq, C // self.n_head)

        y = weight1 + weight2 # B x nh x Tq x hs
        y = y.transpose(1, 2).contiguous() # B x Tq x nh x hs
        y = y.view(B, T, C)
        
        y = self.resid_dropout(self.c_proj(y))
        
        if return_kv:
            return y, (r_k1, r_v1)
        return y

class MLP(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.c_fc    = nn.Linear(config.n_embd, 4 * config.n_embd, bias=config.bias)
        self.gelu    = nn.GELU()
        self.c_proj  = nn.Linear(4 * config.n_embd, config.n_embd, bias=config.bias)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x

class AttentionOnlyBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        return x

class RPETransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = RPESelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x, return_kv=False):
        x = x + self.attn(self.ln_1(x), return_kv=return_kv)
        x = x + self.mlp(self.ln_2(x))
        return x

class TransformerBlock(nn.Module):

    def __init__(self, config):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = MLP(config)
        
    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

@dataclass
class GPTConfig:
    block_size: int = 1024
    vocab_size: int = 50257
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    n_out_embd: int = 768
    dropout: float = 0.0
    bias: bool = True # True: bias in Linears and LayerNorms, like GPT-2. False: a bit better and faster
    use_trained_pe: bool = True
    similarity_function: str = 'cosine'
    attention_block: str = 'AttentionOnlyBlock'
    max_relative_position: int = 2

class GPT(nn.Module):

    def __init__(self, config):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        block = {'TransformerBlock': TransformerBlock,
                 'RPETransformerBlock': RPETransformerBlock}[config.attention_block]

        if config.attention_block == 'RPETransformerBlock':
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ))
        else:
            self.transformer = nn.ModuleDict(dict(
                wte = nn.Embedding(config.vocab_size, config.n_embd),
                wpe = nn.Embedding(config.block_size, config.n_embd) if config.use_trained_pe else PositionalEncoding1D(config.n_embd),
                drop = nn.Dropout(config.dropout),
                h = nn.ModuleList([block(config) for _ in range(config.n_layer)]),
                ln_f = LayerNorm(config.n_embd, bias=config.bias),
            ))
        # self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        self.lm_head = nn.Linear(config.n_embd, config.n_out_embd, bias=False)
        # with weight tying when using torch.compile() some warnings get generated:
        # "UserWarning: functional_call was passed multiple values for tied weights.
        # This behavior is deprecated and will be an error in future versions"
        # not 100% sure what this is, so far seems to be harmless. TODO investigate
        # self.transformer.wte.weight = self.lm_head.weight # https://paperswithcode.com/method/weight-tying
        self.transformer.wte.weight.requires_grad = True
        if config.use_trained_pe and hasattr(self.transformer, 'wpe'):
            self.transformer.wpe.weight.requires_grad = True
        # self.transformer.wte.weight /= torch.norm(self.transformer.wte.weight, dim=1).unsqueeze(1)

        # init all weights
        self.apply(self._init_weights)
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('c_proj.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * config.n_layer))

        # report number of parameters
        print("number of parameters: %.2fM" % (self.get_num_params()/1e6,))

    def get_num_params(self, non_embedding=True):
        """
        Return the number of parameters in the model.
        For non-embedding count (default), the position embeddings get subtracted.
        The token embeddings would too, except due to the parameter sharing these
        params are actually used as weights in the final layer, so we include them.
        """
        n_params = sum(p.numel() for p in self.parameters())
        # comment these if not using learnable pos embedding
        if non_embedding and (hasattr(self.transformer, 'wpe') and isinstance(self.transformer.wpe, nn.Embedding)):
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_tokens, indices=None, indices2=None, targets=None, distance_targets=None, temperature=1.0, lamb_ce=0., lamb_kl=0., lamb_kl_reverse=0., return_kv=False):
        device = input_tokens.device
        b, t = input_tokens.size()
        assert t <= self.config.block_size, f"Cannot forward sequence of length {t}, block size is only {self.config.block_size}"

        if hasattr(self.transformer, 'wpe'):
            pos = torch.arange(0, t, dtype=torch.long, device=device) # shape (t)

            # forward the GPT model itself
            tok_emb = self.transformer.wte(input_tokens) # token embeddings of shape (b, t, n_embd)
            if isinstance(self.transformer.wpe, nn.Embedding):
                pos_emb = self.transformer.wpe(pos) # position embeddings of shape (t, n_embd)
            else:
                pos_emb = self.transformer.wpe(tok_emb) # if using sinusodial, (b, t, n_embed)
            x = self.transformer.drop(tok_emb + pos_emb)
            xv = self.transformer.drop(tok_emb)
            for b_idx, block in enumerate(self.transformer.h):
                if isinstance(block, (ModifiedAttentionOnlyBlock, ModifiedTransformerBlock)):
                    xv = block(x, xv)
                    x = xv + pos_emb if b_idx < len(self.transformer.h) - 1 else xv
                else:
                    x = block(x)
        else:
            # forward the GPT model itself
            tok_emb = self.transformer.wte(input_tokens) # token embeddings of shape (b, t, n_embd)
            x = self.transformer.drop(tok_emb)
            if return_kv:
                all_pastkv = []
            for b_idx, block in enumerate(self.transformer.h):
                if return_kv:
                    x, past_kv = block(x, return_kv=return_kv)
                    all_pastkv.append(past_kv)
                else:
                    x = block(x, return_kv=return_kv)
        next_emb = self.transformer.ln_f(x) # next token embeddings of shape (b, t, n_embd)

        if targets is not None or distance_targets is not None:
            # if we are given some desired targets also calculate the loss
            logits = self.lm_head(next_emb)
            logits = logits.view(-1, logits.size(-1))
            if indices2 is not None:
                assert indices is not None
                logits1 = logits[indices]
                logits2 = logits[indices2] #.detach()
                if self.config.similarity_function == 'cosine':
                    logits1, logits2 = F.normalize(logits1, dim=-1), F.normalize(logits2, dim=-1)
                    sim_matrix = logits1 @ logits2.T
                    similarity_all = similarity = sim_matrix / temperature
                    distance_all = distance = 1 / temperature - similarity_all
                elif self.config.similarity_function == 'jsd':
                    log_probs1 = logits1.log_softmax(dim=-1)
                    log_probs2 = logits2.log_softmax(dim=-1)
                    distance = (log_probs2[None, :].exp() * (log_probs2[None, :] - log_probs1[:, None].detach())).sum(dim=-1)
                    distance += (log_probs1[:, None].exp() * (log_probs1[:, None] - log_probs2[None, :].detach())).sum(dim=-1)
                    distance = distance / 2
                    distance_all = distance
                    similarity_all = similarity = -distance
                elif self.config.similarity_function == 'l2':
                    distance = ((logits2[None, :] - logits1[:, None]) ** 2).mean(dim=-1)
                    distance_all = distance
                    similarity_all = similarity = -distance
            else:
                if indices is not None:
                    logits = logits[indices]
                if self.config.similarity_function == 'cosine':
                    logits = F.normalize(logits, dim=-1)
                    mask = torch.eye(logits.shape[0], dtype=torch.bool)
                    sim_matrix = logits @ logits.T
                    similarity_all = sim_matrix / temperature
                    distance_all = 1 / temperature - similarity_all
                    sim_matrix = sim_matrix[~mask].view(-1, sim_matrix.shape[0] - 1)
                    similarity = (1 - sim_matrix) / temperature
                    distance = -similarity
                elif self.config.similarity_function == 'jsd':
                    log_probs = logits.log_softmax(dim=-1)
                    distance = (log_probs[None, :].exp() * (log_probs[None, :] - log_probs[:, None].detach())).sum(dim=-1)
                    distance += (log_probs[:, None].exp() * (log_probs[:, None] - log_probs[None, :].detach())).sum(dim=-1)
                    distance = distance / 2
                    distance_all = distance
                    similarity_all = -distance
                    mask = torch.eye(distance.shape[0], dtype=torch.bool)
                    distance = distance[~mask].view(-1, distance.shape[0] - 1)
                    similarity = -distance
                elif self.config.similarity_function == 'l2':
                    distance = ((logits[None, :] - logits[:, None]) ** 2).mean(dim=-1)
                    distance_all = distance
                    similarity_all = -distance
                    mask = torch.eye(distance.shape[0], dtype=torch.bool)
                    distance = distance[~mask].view(-1, distance.shape[0] - 1)
                    similarity = -distance
            
            loss = 0
            if lamb_ce > 0:
                loss_ce = F.cross_entropy(similarity, targets)
                loss += lamb_ce * loss_ce
            if lamb_kl > 0:
                loss_kl = F.kl_div((-distance_all).log_softmax(dim=-1), (-distance_targets).log_softmax(dim=-1), log_target=True, reduction='batchmean')
                loss += lamb_kl * loss_kl
            if lamb_kl_reverse > 0:
                mask = (distance_targets.sum(0) == distance_targets.sum(0))
                loss_kl_reverse = F.kl_div((-distance_targets[:, mask]).log_softmax(dim=-1), (-distance_all[:, mask]).log_softmax(dim=-1), log_target=True, reduction='batchmean')
                loss += lamb_kl_reverse * loss_kl_reverse
        else:
            # inference-time mini-optimization: only forward the lm_head on the very last position
            logits = self.lm_head(next_emb) # note: using list [-1] to preserve the time dim
            distance = None
            loss = None
            if self.config.similarity_function == 'cosine':
                logits = F.normalize(logits, dim=-1)

        if return_kv:
            return logits, distance, loss, all_pastkv
        return logits, distance, loss

    @torch.no_grad()
    def get_distance(self, input_tokens, indices1, indices2, temperature=0.1, split_size=512):
        all_logits = []
        for _input_tokens in input_tokens.split(split_size):
            logits, _, _ = self.forward(_input_tokens)
            all_logits.append(logits)
        logits = torch.cat(all_logits, dim=0)
        B, L, D = indices1.shape
        logits = logits.view(B, L, D, -1)
        logits1 = logits[indices1].view(B, -1, logits.size(-1))
        logits2 = logits[indices2].view(B, -1, logits.size(-1))
        if self.config.similarity_function == 'cosine':
            logits1 = F.normalize(logits1, dim=-1)
            logits2 = F.normalize(logits2, dim=-1)
            sim_matrix = torch.einsum('bij,bkj->bik', logits1, logits2)
            distance = (1 - sim_matrix) / temperature
        elif self.config.similarity_function == 'jsd':
            log_probs1 = logits1.log_softmax(dim=-1)
            log_probs2 = logits2.log_softmax(dim=-1)
            torch.empy_cache()
            distance = (log_probs1[:, :, None].exp() * (log_probs1[:, :, None] - log_probs2[:, None])).sum(dim=-1)
            torch.empy_cache()
            distance += (log_probs2[:, None].exp() * (log_probs2[:, None] - log_probs1[:, :, None])).sum(dim=-1)
            distance = distance / 2
        elif self.config.similarity_function == 'l2':
            distance = ((logits1[:, :, None] - logits2[:, None]) ** 2).mean(dim=-1)
        return distance, logits2
    
    def crop_block_size(self, block_size):
        # model surgery to decrease the block size if necessary
        # e.g. we may load the GPT2 pretrained model checkpoint (block size 1024)
        # but want to use a smaller block size for some smaller, simpler model
        assert block_size <= self.config.block_size
        self.config.block_size = block_size
        self.transformer.wpe.weight = nn.Parameter(self.transformer.wpe.weight[:block_size])
        for block in self.transformer.h:
            if hasattr(block.attn, 'bias'):
                block.attn.bias = block.attn.bias[:,:,:block_size,:block_size]

    @classmethod
    def from_pretrained(cls, model_type, override_args=None):
        assert model_type in {'gpt2', 'gpt2-medium', 'gpt2-large', 'gpt2-xl'}
        override_args = override_args or {} # default to empty dict
        # only dropout can be overridden see more notes below
        assert all(k != 'n_embd' and k != 'vocab_size' for k in override_args)
        from transformers import GPT2LMHeadModel
        print("loading weights from pretrained gpt: %s" % model_type)

        # n_layer, n_head and n_embd are determined from model_type
        config_args = {
            'gpt2':         dict(n_layer=12, n_head=12, n_embd=768),  # 124M params
            'gpt2-medium':  dict(n_layer=24, n_head=16, n_embd=1024), # 350M params
            'gpt2-large':   dict(n_layer=36, n_head=20, n_embd=1280), # 774M params
            'gpt2-xl':      dict(n_layer=48, n_head=25, n_embd=1600), # 1558M params
        }[model_type]
        # we can override the dropout rate, if desired
        for k, v in override_args.items():
            config_args[k] = override_args[k]
        # create a from-scratch initialized minGPT model
        config = GPTConfig(**config_args)
        model = cls(config,)
        sd = model.state_dict()

        # init a huggingface/transformers model
        model_hf = GPT2LMHeadModel.from_pretrained(model_type)
        sd_hf = model_hf.state_dict()

        sd_keys_hf = sd_hf.keys()
        copy_sd_keys_hf = [] 
        copy_sd_keys_hf += [k for k in sd_keys_hf if k.endswith('.wte.weight')] # ignore these, just a buffer
        if config.use_trained_pe:
            copy_sd_keys_hf += [k for k in sd_keys_hf if k.endswith('.wpe.weight')] # ignore these, just a buffer

        for k in copy_sd_keys_hf:
            try:
                assert sd_hf[k].shape == sd[k].shape
                with torch.no_grad():
                    sd[k].copy_(sd_hf[k])
            except:
                end_k = k.replace(k.split('.')[0] + '.', '')
                for _k in sd.keys():
                    if _k.endswith(end_k) and sd_hf[k].shape == sd[_k].shape:
                        with torch.no_grad():
                            sd[_k].copy_(sd_hf[k])
                            print(f"copying {k} to {_k}")
                            break
        del model_hf
        return model

    def configure_optimizers(self, weight_decay, learning_rate, betas, device_type):
        # start with all of the candidate parameters
        param_dict = {pn: p for pn, p in self.named_parameters()}
        # filter out those that do not require grad
        param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
        # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
        # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]
        num_decay_params = sum(p.numel() for p in decay_params)
        num_nodecay_params = sum(p.numel() for p in nodecay_params)
        print(f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters")
        print(f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters")
        # Create AdamW optimizer and use the fused version if it is available
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()
        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"using fused AdamW: {use_fused}")

        return optimizer

    def estimate_mfu(self, fwdbwd_per_iter, dt):
        """ estimate model flops utilization (MFU) in units of A100 bfloat16 peak FLOPS """
        # first estimate the number of flops we do per iteration.
        # see PaLM paper Appendix B as ref: https://arxiv.org/abs/2204.02311
        N = self.get_num_params()
        cfg = self.config
        L, H, Q, T = cfg.n_layer, cfg.n_head, cfg.n_embd//cfg.n_head, cfg.block_size
        flops_per_token = 6*N + 12*L*H*Q*T
        flops_per_fwdbwd = flops_per_token * T
        flops_per_iter = flops_per_fwdbwd * fwdbwd_per_iter
        # express our flops throughput as ratio of A100 bfloat16 peak flops
        flops_achieved = flops_per_iter * (1.0/dt) # per second
        flops_promised = 312e12 # A100 GPU bfloat16 peak flops is 312 TFLOPS
        mfu = flops_achieved / flops_promised
        return mfu

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx