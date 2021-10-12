import math

import numpy as np
from munch import Munch
from torch.utils.data import DataLoader

from exp import ex
from utils import merge_dict
from .dataset import get_dataset

@ex.capture()
def get_dataloaders(batch_size, grad_acc_steps, num_workers, max_epochs, modes=['train', 'val',' test']):
    dataset, video, tokenizer = get_dataset(modes=modes)
    outputs = {}

    for mode, ds in dataset.items():
        dataloader = DataLoader(ds,
                                batch_size=batch_size,
                                collate_fn=ds.collate_fn,
                                shuffle=(mode == 'train' or mode == 'pretrain'),
                                num_workers=num_workers)
        dataloader.dataset.t_total = math.ceil(len(ds) * max_epochs / (batch_size * grad_acc_steps))
        outputs[mode] = dataloader
    return outputs, video, tokenizer
