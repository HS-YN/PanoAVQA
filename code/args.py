import re
import json
import random
from datetime import datetime
from pathlib import Path, PosixPath

import torch
import numpy as np
from munch import Munch
from sacred.arg_parser import get_config_updates

from config import args_dict
from model.config import *


def get_args(options, fixed_args={}):
    '''Processes arguments'''
    updated_args = {}
    updated_args.update(get_new_args(options))
    updated_args.update(fixed_args)

    default_args = get_default_config(options, fixed_args)
    #import json; print(json.dumps(current_args, indent=2))
    #assert False

    args = Munch(default_args)
    args.update(Munch(updated_args))
    if args.ckpt_name is not None:
        args.update(Munch(load_args(args)))
        args.update(Munch(updated_args))
    
    args.update(fix_seed(args))
    args.update(resolve_paths(args))
    args = update_data_path(args)

    args.config_dir = get_config_dir(args)
    args.model_config = get_model_config(args.model_name, args.model_config)
    args = args.toDict()

    # Primary assertions
    if args['device'] == 'cuda':
        assert torch.cuda.is_available(), "GPU device is not available"

    return args


def get_default_config(options, fixed_args):
    updated_args = get_config_updates(options['UPDATE'])[0]
    if 'model_name' in updated_args.keys() and updated_args['model_name'] in args_dict.keys():
        return args_dict[updated_args['model_name']]
    else:
        return default_args


def get_model_config(model_name, args):
    model_config = ModelConfig(**args)
    # Sacred is not capable of capturing classmethods as variable
    return {k:v for k,v in vars(model_config).items()}


def get_new_args(options):
    '''Fetch updated arguments that deviate from default settings'''
    if 'UPDATE' in options:
        new_args, _ = get_config_updates(options['UPDATE'])
    else:
        new_args = options
    return new_args


def load_args(args):
    '''Load arguments of previous experiment'''
    root = Path('../').resolve()

    if str(root) not in str(args.ckpt_path):
        args.ckpt_path = root / args.ckpt_path
    args_path = sorted(args.ckpt_path.glob(f'{args.ckpt_name}*'))
    if args.ckpt_name is None or len(args_path) <= 0:
        return {}
    args_path = args_path[0] / 'args.json'
    ckpt_args = {}
    if args_path.is_file():
        ckpt_args = json.load(open(args_path, 'r'))['args']
        # update non-string arguments (and data_path)
        eval_keys = [k for k, v in default_args.items() if not isinstance(v, str)]
        eval_keys.append('data_path')
        # ckpt_args = {k: eval(v) if k in eval_keys else v for k, v in ckpt_args.items()}
        ckpt_args = {k: v for k, v in ckpt_args.items() if not k.endswith('_path')}
    return ckpt_args


def resolve_paths(args):
    '''Convert strings into paths if applicable'''
    path_list = [k for k in args.keys() if k.endswith('_path') and k != 'data_path']
    res_args = {}
    res_args['root'] = Path('../').resolve()
    for path in path_list:
        if args[path] is not None:
            if isinstance(args[path], list):
                res_args[path] = [res_args['root'] / Path(v) for v in args[path]]
            elif isinstance(args[path], dict):
                res_args[path] = {k: res_args['root'] / Path(v) for k, v in args[path].items()}
            else:
                res_args[path] = res_args['root'] / Path(args[path])
    return res_args


def update_data_path(args):
    '''Update dataset path'''
    if 'data_path' not in args:
        args['data_path'] = {}
    for k in ['pretrain', 'train', 'preval', 'val', 'test']:
        if f"{k}_path" in args:
            args['data_path'][k] = args[f"{k}_path"]
            del args[f"{k}_path"]
    for k, path in args.data_path.items():
        path = Path(path).resolve() if str(path).startswith('/') else args.root / path
        args.data_path[k] = path
    return args



def fix_seed(args):
    '''Fix random seeds at once'''
    if 'random_seed' not in args or not isinstance(args['random_seed'], int):
        args['random_seed'] = args['seed'] if 'seed' in args else 0
    args['seed'] = args['random_seed'] # for sacred

    random.seed(args['random_seed'])
    np.random.seed(args['random_seed'])
    torch.manual_seed(args['random_seed'])
    torch.cuda.manual_seed_all(args['random_seed'])
    # torch.backends.cudnn.benchmark = False
    # torch.set_deterministic(True)
    torch.multiprocessing.set_sharing_strategy('file_system')

    return args


def get_config_dir(args):
    '''Generate directory name for logging'''
    now = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    tags = [re.sub('[,\W-]+', '_', str(args[key])) for key in args['log_keys']]
    dirname = '_'.join(tags)[:100] # Avoid too long paths
    return f"{dirname}_{now}"
