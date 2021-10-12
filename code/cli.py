import json

from exp import ex
from args import get_args

from train import _train
from evaluate import _eval

@ex.command
def train(_config):
    res = _train()
    print("Training complete")

    return 0


@ex.command
def eval(_config):
	res = _eval()
	print("Evaluation complete")

	return 0


@ex.option_hook
def update_args(options):
    args = get_args(options)
    print(json.dumps({k: str(v) for k, v in sorted(args.items())}, indent=4))
    ex.add_config(args)
    return options


@ex.automain
def run():
    train()