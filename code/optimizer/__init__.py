import os
from pathlib import Path

from torch import optim
from inflection import underscore

from exp import ex
from .schedulers import get_scheduler


optim_dict = {}


def add_optims():
    path = Path(os.path.dirname(__file__))

    for p in path.glob('*.py'):
        name = p.stem
        parent = p.parent.stem

        if name != "__init__":
            __import__("{}.{}".format(parent, name))
            module = eval(name)
            for member in dir(module):
                member = getattr(module, member)
                if hasattr(member, '__bases__') and \
                        (optim.Optimizer in member.__bases__ or \
                            optim.lr_scheduler._LRScheduler in member.__bases__ or \
                            optim.Optimizer in member.__bases__[0].__bases__ or \
                            optim.lr_scheduler._LRScheduler in member.__bases[0].__bases__):
                    optim_dict[underscore(str(member.__name__))] = member


@ex.capture()
def get_optimizer(model, t_total, optimizer_name, learning_rate):
    optim = optim_dict[optimizer_name](model, learning_rate)
    optim.zero_grad()
    scheduler = get_scheduler(optim, t_total)
    return optim, scheduler


add_optims()