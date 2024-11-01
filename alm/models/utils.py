from scipy.sparse import coo_array


def get_sparse_array_from_result(result_by_token_id, vocab_size):
    indices, values = [], []
    for k, v in result_by_token_id.items():
        indices.append(k)
        values.append(v['cont_cnt'])
    return coo_array((values, (indices, [0] * len(indices))), shape=(vocab_size, 1))