import shutil
import subprocess as sp
from pathlib import Path

import torch
import torch.nn as nn

from exp import ex


class BertPreTrainedModel(nn.Module):
    def __init__(self, model_config, cache_path, *inputs, **kwargs):
        super().__init__()
        self.config = model_config
        self.cache_path = cache_path

    def init_bert_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

    @classmethod
    def from_pretrained(cls, model_config, cache_path, pretrained_model='bert-base-uncased', 
                        state_dict=None, from_tf=False, *inputs, **kwargs):
        # Assume bert-base-uncased
        assert pretrained_model == 'bert-base-uncased', f"[ERROR] {pretrained_model} is not supported."

        pretrained_model_path = cache_path / pretrained_model
        if not pretrained_model_path.is_dir():
            pretrained_model_path.mkdir(parents=True, exist_ok=True)

        pretrained_model = pretrained_model_path / 'pytorch_model.bin'
        if not Path(pretrained_model).exists():
            print("[LOG] Downloading BERT pretrained weight. It will take somewhere around 10 minutes, depending on your internet connection.")
            sp.call(["wget", "-P", f"{pretrained_model_path}",
                     "https://s3.amazonaws.com/models.huggingface.co/bert/bert-base-uncased.tar.gz"])
            sp.call(["tar", "-xvf", "{}".format(pretrained_model_path / 'bert-base-uncased.tar.gz')])
            (pretrained_model_path / 'bert-base-uncased.tar.gz').unlink()

        model = cls(model_config, cache_path, *inputs, **kwargs)

        state_dict = torch.load(pretrained_model, map_location='cpu' if not torch.cuda.is_available() else None)

        old_keys = []
        new_keys = []
        for key in state_dict.keys():
            new_key = None
            if "gamma" in key:
                new_key = key.replace('gamma', 'weight')
            if "beta" in key:
                new_key = key.replace('beta', 'bias')
            if new_key:
                old_keys.append(key)
                new_keys.append(new_key)
        for old_key, new_key in zip(old_keys, new_keys):
            state_dict[new_key] = state_dict.pop(old_key)

        missing_keys = []
        unexpected_keys = []
        error_msgs = []
        metadata = getattr(state_dict, '_metadata', None)
        state_dict = state_dict.copy()
        if metadata is not None:
            state_dict._metadata = metadata

        def load(module, prefix=''):
            local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
            module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                         missing_keys, unexpected_keys, error_msgs)
            for name, child in module._modules.items():
                if child is not None:
                    load(child, prefix + name + '.')
        start_prefix = ''
        if not hasattr(model, 'bert') and any(s.startwith('bert.') for s in state_dict.keys()):
            start_prefix = 'bert.'
        load(model, prefix=start_prefix)

        # print('\n'.join(['{} {}'.format(k, v.size()) for k,v in state_dict.items()]))
        # print("Missing Keys: ", '\n'.join(missing_keys))
        # print("Unexpected Keys: ", '\n'.join(unexpected_keys))

        if len(error_msgs) > 0:
            raise RuntimeError('Error(s) in loading state_dict for {}:\n\t{}'.format(
                               model.__class__.__name__, "\n\t".join(error_msgs)))
        return model