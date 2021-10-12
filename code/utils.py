import torch
import numpy as np


def merge_dict(list_of_dict):
    res = {}
    if len(list_of_dict) > 0:
        keys = list_of_dict[0].keys()
        for key in keys:
            res[key] = [d[key] for d in list_of_dict]
    return res


def one_hot_vectorize(target, element_list):
    vector = [1 if ele == target else 0 for ele in element_list]
    
    if sum(vector) == 0: ## target is not in the list
        vector += [1]
    else:
        vector += [0]
    
    return np.array(vector)
