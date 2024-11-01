from infini_gram.engine import InfiniGramEngine
import torch
import numpy as np
import requests
from scipy.sparse import coo_array

from .build_infinigram_py import InfiniGramModelPython
from .induction_only import InductionOnly


def get_sparse_array_from_result(result_by_token_id, vocab_size, key='cont_cnt'):
    indices, values = [], []
    for k, v in result_by_token_id.items():
        indices.append(k)
        values.append(v[key])
    return coo_array((values, (indices, [0] * len(indices))), shape=(vocab_size, 1))


class InductionGram(InductionOnly):
    '''Class that fits Induction-Gram
    '''

    def __init__(
        self,
        tokenizer,
        infinigram_checkpoints=['v4_pileval_gpt2'],
        infinigram_context_length=500,
        fuzzy_mm_name=None,
        fuzzy_context_window_length=64,
        device='cuda',
        random_state=42,
        load_to_ram=False,
    ):
        super(InductionGram, self).__init__(tokenizer, fuzzy_mm_name, fuzzy_context_window_length, device, random_state)

        # set parameters
        self.tokenizer_ = tokenizer
        self.infinigram_context_length = infinigram_context_length
        self.infinigram_checkpoints = infinigram_checkpoints

        # initialize model parameters
        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)
        self.vocab_size_ = len(self.tokenizer_)
        infinigram_engines = []
        for ckpt in infinigram_checkpoints:
            try:
                infinigram_engines.append(InfiniGramEngine(load_to_ram=load_to_ram, index_dir=ckpt, eos_token_id=self.tokenizer_.eos_token_id))
            except:
                infinigram_engines.append(ckpt.split('/')[-1])
                print(f"Fail to load local indexes, so use API endpoint: {ckpt.split('/')[-1]}")
        self.infinigram_engines = infinigram_engines

    def infgram_ntd(self, engine, input_ids):
        if isinstance(engine, str):
            payload = {
                'index': engine,
                'query_type': 'infgram_ntd',
                'query_ids': input_ids,
            }
            results = requests.post('https://api.infini-gram.io/', json=payload).json()
            if 'suffix_len' not in results:
                print(results)
        else:
            results = engine.infgram_ntd(prompt_ids=input_ids)
        return results

    def predict_prob(self, batch_input_ids, suffix_len_thres=9, split_size=16, return_others=False):
        batch = True
        if isinstance(batch_input_ids, torch.Tensor) and batch_input_ids.ndim == 1:
            batch_input_ids = batch_input_ids.unsqueeze(0)
            batch = False
        elif isinstance(batch_input_ids, (list, np.ndarray)) and isinstance(batch_input_ids[0], int):
            batch_input_ids = [batch_input_ids]
            batch = False
        
        # inference Induction-only (exact)
        all_exact_cnt, all_others, all_exact_suffix_len = [], [] ,[]
        for input_ids in batch_input_ids:
            input_ids = input_ids.tolist() if not isinstance(input_ids, list) else input_ids

            # Infini-Gram
            suffix_len = -1
            for engine in self.infinigram_engines:
                results = self.infgram_ntd(engine, input_ids[-self.infinigram_context_length:])
                if results['suffix_len'] > suffix_len:
                    result_by_token_id = results['result_by_token_id']
                    suffix_len, prompt_cnt = results['suffix_len'], results['prompt_cnt']
                elif results['suffix_len'] == suffix_len:
                    for k, v in results['result_by_token_id'].items():
                        if k in result_by_token_id.keys():
                            result_by_token_id[k]['cont_cnt'] += v['cont_cnt']
                        else:
                            result_by_token_id[k] = v
                    prompt_cnt += results['prompt_cnt']

            reference_cnt = get_sparse_array_from_result(result_by_token_id, self.vocab_size_)
            reference_cnt = reference_cnt.toarray()[:, 0]

            # Induction-only (exact)
            incontext_lm = InfiniGramModelPython.from_data(documents_tkn=input_ids[:-1], tokenizer=self.tokenizer_)
            prob_next_distr = incontext_lm.predict_prob(np.array(input_ids))
            if prob_next_distr.effective_n >= suffix_len:
                suffix_len = prob_next_distr.effective_n
                incontext_cnt = prob_next_distr.count
                all_exact_cnt.append(incontext_cnt)
                incontext_sparse = (len(incontext_cnt.nonzero()[0]) == 1)
                if return_others:
                    all_others.append({'suffix_len': suffix_len, 'prompt_cnt': incontext_cnt.sum(), 'sparse': incontext_sparse, 'model': 'exact-induction'})
            else:
                all_exact_cnt.append(reference_cnt)
                if return_others:
                    all_others.append({'suffix_len': suffix_len, 'prompt_cnt': prompt_cnt, 'sparse': (len(reference_cnt.nonzero()[0]) == 1), 'model': 'infinigram'})
            all_exact_suffix_len.append(suffix_len)

        all_final_cnt = np.stack(all_exact_cnt, axis=0).astype(np.float32)
        all_exact_suffix_len = np.array(all_exact_suffix_len)

        fuzzy_round_mask = all_exact_suffix_len < suffix_len_thres
        if fuzzy_round_mask.any():
            all_fuzzy_cnt, weight = self._fuzzy_matching_in_context(batch_input_ids[fuzzy_round_mask], split_size=split_size)
            for idx in fuzzy_round_mask.nonzero()[0]:
                all_others[idx]['model'] = 'fuzzy-induction'
            all_final_cnt[fuzzy_round_mask] = all_fuzzy_cnt
        all_final_probs = all_final_cnt / all_final_cnt.sum(1, keepdims=True)

        if batch:
            if return_others:
                return all_final_probs, all_others
            return all_final_probs
        else:
            if return_others:
                return all_final_probs[0], all_others[0]
            return all_final_probs[0]
