from torch.optim.lr_scheduler import LambdaLR
from transformers import get_linear_schedule_with_warmup

from exp import ex


def get_no_scheduler(optimizer, num_warmup_steps, num_training_steps):
    def lr_lambda(current_step):
        return 1

    return LambdaLR(optimizer, lr_lambda)


sched_dict = {
    'linear': get_linear_schedule_with_warmup,
    'none': get_no_scheduler
}


@ex.capture()
def get_scheduler(optimizer, t_total, warmup, scheduler_name, grad_acc_steps):
    warmup_steps = int(t_total * warmup)
    scheduler = sched_dict[scheduler_name](optimizer, warmup_steps, t_total)
    scheduler.accumulated = 0
    scheduler.grad_acc_steps = grad_acc_steps
    return scheduler