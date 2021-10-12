import torch
import numpy as np
from torch import nn

from exp import ex
from ckpt import load_ckpt
from model import get_model
from data.dataloader import get_dataloaders


@ex.capture()
def prepare_batch(batch, device):

    data, label, meta = batch #zip(*batch)

    for key, value in data.items():
        if isinstance(value, list):
            data[key] = [convert(v, device) for v in value]
        elif isinstance(value, dict):
            data[key] = {k: convert(v, device) for k, v in value.items()}
        else:
            data[key] = convert(value, device)

    for key, value in label.items():
        if isinstance(value, list):
            label[key] = [convert(v, device) for v in value]
        elif isinstance(value, dict):
            label[key] = {k: convert(v, device) for k, v in value.items()}
        else:
            label[key] = convert(value, device)

    return data, label, meta


def convert(value, device):
    if isinstance(value, np.ndarray):
        value = torch.from_numpy(value)
    if torch.is_tensor(value):
        value = value.to(device)
    return value


@ex.capture()
def get_all(data_modes, device):
    dataloaders, video, tokenizer = get_dataloaders(modes=data_modes)
    model = get_model(tokenizer)
    model = load_ckpt(model).to(device)
    criterion = get_criterion()

    return dataloaders, video, tokenizer, model, criterion


def get_criterion():
    criterion = nn.CrossEntropyLoss()
    return criterion
