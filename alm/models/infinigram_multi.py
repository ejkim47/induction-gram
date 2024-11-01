from infini_gram.engine import InfiniGramEngine
import torch
import numpy as np
import requests
from alm.models.utils import get_sparse_array_from_result


class InfiniGramMulti:
    '''Class that fits Infini-Gram
    '''

    def __init__(
        self,
        tokenizer,
        infinigram_checkpoints=['v4_pileval_gpt2'],
        infinigram_context_length=5,
        random_state=42,
        load_to_ram=False,
    ):

        # set parameters
        self.tokenizer_ = tokenizer
        self.infinigram_checkpoints = infinigram_checkpoints
        self.infinigram_context_length = infinigram_context_length
        self.random_state = random_state

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

    def predict_prob(self, batch_input_ids, return_others=False):
        batch = True
        if isinstance(batch_input_ids, torch.Tensor) and batch_input_ids.ndim == 1:
            batch_input_ids = batch_input_ids.unsqueeze(0)
            batch = False
        elif isinstance(batch_input_ids, (list, np.ndarray)) and isinstance(batch_input_ids[0], int):
            batch_input_ids = [batch_input_ids]
            batch = False
        all_infinigram_probs, all_others = [], []
        for input_ids in batch_input_ids:
            input_ids = input_ids.tolist() if not isinstance(input_ids, list) else input_ids
            suffix_len = -1
            for engine in self.infinigram_engines:
                cropped_input_ids = input_ids[-self.infinigram_context_length:] if len(input_ids) > self.infinigram_context_length else input_ids
                results = self.infgram_ntd(engine, cropped_input_ids)
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

            infinigram_cnt = get_sparse_array_from_result(result_by_token_id, self.vocab_size_)
            sparse = (len(infinigram_cnt.nonzero()[0]) == 1)
            infinigram_cnt = infinigram_cnt.toarray()[:, 0]
            infinigram_probs = infinigram_cnt / infinigram_cnt.sum()
            others = {'suffix_len': suffix_len, 'prompt_cnt': prompt_cnt, 'sparse': sparse}
            all_infinigram_probs.append(infinigram_probs)
            all_others.append(others)

        if batch:
            if return_others:
                return all_infinigram_probs, all_others
            return all_infinigram_probs
        else:
            if return_others:
                return all_infinigram_probs[0], all_others[0]
            return all_infinigram_probs[0]