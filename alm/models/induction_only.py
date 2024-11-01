from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import numpy as np
import alm.config
from .utils import get_sparse_array_from_result

from .build_infinigram_py import InfiniGramModelPython
from .mini_gpt import GPTConfig, GPT


class InductionOnly:
    '''Class that fits Induction-only (exact/fuzzy)
    '''

    def __init__(
        self,
        tokenizer,
        fuzzy_mm_name=None,
        fuzzy_context_window_length=64,
        device='cuda',
        random_state=42,
    ):
        
        # set parameters
        self.tokenizer_ = tokenizer
        self.fuzzy_context_window_length = fuzzy_context_window_length
        self.device = device
        self.random_state = random_state

        if fuzzy_mm_name is not None:
            self._load_fuzzy_mm(fuzzy_mm_name)

        # initialize model parameters
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.vocab_size_ = len(self.tokenizer_)

    def _load_fuzzy_mm(self, fuzzy_mm_name):
        if not fuzzy_mm_name.endswith('.pt'):
            self.fuzzy_mm = AutoModelForCausalLM.from_pretrained(fuzzy_mm_name, token=alm.config.TOKEN_HF, load_in_8bit=False, device_map="auto").eval()
            self.use_llm_as_fuzzy = True
        else:
            checkpoint = torch.load(fuzzy_mm_name, map_location=self.device)
            # create the model
            gptconf = GPTConfig(**checkpoint['model_kwargs'])
            fuzzy_mm = GPT(gptconf)
            state_dict = checkpoint['model']
            # fix the keys of the state dictionary :(
            # honestly no idea how checkpoints sometimes get this prefix, have to debug more
            unwanted_prefix = 'module.'
            for k, v in list(state_dict.items()):
                if k.startswith(unwanted_prefix):
                    state_dict[k[len(unwanted_prefix):]] = state_dict.pop(k)
            fuzzy_mm.load_state_dict(state_dict)
            fuzzy_mm.to(self.device)
            self.fuzzy_mm = fuzzy_mm
            print(f"Loaded model from {fuzzy_mm_name} (saved at iter {checkpoint['iter_num']})")

            self.use_llm_as_fuzzy = False

    @torch.no_grad()
    def _fuzzy_matching_in_context(self, batch_input_ids, split_size=16):
        B = len(batch_input_ids)
        if self.fuzzy_context_window_length > batch_input_ids.shape[-1]:
            input_ids_tensor = torch.Tensor(batch_input_ids).unsqueeze(1).long()
        else:
            input_ids_tensor = batch_input_ids.unfold(1, self.fuzzy_context_window_length, 1)
        
        if self.use_llm_as_fuzzy:
            all_logits = []
            if B == 1:
                for i, _batch_input_ids in enumerate(input_ids_tensor.view(-1, input_ids_tensor.shape[-1]).split(split_size)):
                    logits = self.fuzzy_mm(_batch_input_ids.to(self.device)).logits
                    if i != 0:
                        logits = logits[:, -1:]
                    all_logits.append(logits)
                logits = torch.cat(all_logits, dim=0).unsqueeze(0)
            else:
                for i, _batch_input_ids in enumerate(input_ids_tensor.permute(1, 0, 2)):
                    logits = self.fuzzy_mm(_batch_input_ids.to(self.device)).logits
                    if i != 0:
                        logits = logits[:, -1:]
                    all_logits.append(logits)
                logits = torch.cat(all_logits, dim=1)
            torch.cuda.empty_cache()
            log_probs = logits.log_softmax(dim=-1)
            distance = (log_probs[:, :-1].exp() * (log_probs[:, :-1] - log_probs[:, -1:])).sum(dim=-1).cpu().detach()
            distance += (log_probs[:, -1:].exp() * (log_probs[:, -1:] - log_probs[:, :-1])).sum(dim=-1).cpu().detach()
            distance = distance / 2
        else:
            indices1 = torch.zeros_like(input_ids_tensor).bool()
            indices2 = torch.zeros_like(input_ids_tensor).bool()
            indices1[:, 0, :-1] = True
            indices1[:, :-1, -1] = True
            indices2[:, -1, -1] = True
            input_ids_tensor = input_ids_tensor.contiguous().view(-1, input_ids_tensor.shape[-1])
            distance, logits = self.fuzzy_mm.get_distance(input_ids_tensor.to(self.device), indices1, indices2, temperature=0.1, split_size=split_size)
            distance = distance.cpu()[..., 0] # len(input_ids) - 1
        weight = (-distance).exp()
        batch_ids = torch.arange(B).view(-1, 1).repeat(1, weight.shape[-1]).view(-1).to(batch_input_ids.device)
        input_ids = batch_input_ids[:, -weight.shape[-1]:].reshape(-1)
        cnt = torch.sparse_coo_tensor(torch.stack([batch_ids, input_ids], dim=0), weight.view(-1), size=(B, self.vocab_size_)).to_dense()
        return np.array(cnt), weight

    @torch.no_grad()
    def predict_prob(self, batch_input_ids, incontext_mode='exact', split_size=16, return_others=False):
        batch = True
        if isinstance(batch_input_ids, torch.Tensor) and batch_input_ids.ndim == 1:
            batch_input_ids = batch_input_ids.unsqueeze(0)
            batch = False
        elif isinstance(batch_input_ids, (list, np.ndarray)) and isinstance(batch_input_ids[0], int):
            batch_input_ids = [batch_input_ids]
            batch = False
        if incontext_mode == 'exact': # do not work as batch
            all_incontext_cnt, others = [], []
            for input_ids in batch_input_ids:
                input_ids = input_ids.tolist() if not isinstance(input_ids, list) else input_ids
                incontext_lm = InfiniGramModelPython.from_data(documents_tkn=input_ids[:-1], tokenizer=self.tokenizer_)
                prob_next_distr = incontext_lm.predict_prob(np.array(input_ids))
                incontext_cnt = prob_next_distr.count
                suffix_len = prob_next_distr.effective_n
                incontext_sparse = (len(incontext_cnt.nonzero()[0]) == 1)
                all_incontext_cnt.append(incontext_cnt)
                others.append({'suffix_len': suffix_len, 'prompt_cnt': incontext_cnt.sum(), 'sparse': incontext_sparse})
            all_incontext_cnt = np.stack(all_incontext_cnt, axis=0)
            all_incntext_probs = all_incontext_cnt / all_incontext_cnt.sum(1, keepdims=True)
            all_incntext_probs = np.where(all_incntext_probs != all_incntext_probs, np.ones_like(all_incntext_probs) / all_incntext_probs.shape[1], all_incntext_probs) # prevent inf probability
        elif incontext_mode == 'fuzzy': # work as batch
            assert hasattr(self, 'fuzzy_mm')
            all_incontext_cnt, weight = self._fuzzy_matching_in_context(batch_input_ids, split_size=split_size)
            all_incntext_probs = all_incontext_cnt / all_incontext_cnt.sum(1, keepdims=True)
            all_incntext_probs = np.where(all_incntext_probs != all_incntext_probs, np.ones_like(all_incntext_probs) / all_incntext_probs.shape[1], all_incntext_probs) # prevent inf probability
            if return_others:
                others = [{'suffix_len': 0, 'prompt_cnt': incontext_cnt.sum(), 'sparse': (len(incontext_cnt.nonzero()[0]) == 1), 'weight': _weight} for incontext_cnt, _weight in zip(all_incontext_cnt, weight)]
        else:
            raise ValueError(f"Unknown incontext_mode: {incontext_mode}")

        if batch:
            if return_others:
                return all_incntext_probs, others
            return all_incntext_probs
        else:
            if return_others:
                return all_incntext_probs[0], others[0]
            return all_incntext_probs[0]
