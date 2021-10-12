from torch.optim import Adam
from transformers import AdamW

from exp import ex


class BertAdam(AdamW):
    @ex.capture()
    def __init__(self, model, learning_rate, transformer_learning_rate, weight_decay):
        options = {}
        options['lr'] = learning_rate
        options['weight_decay'] = weight_decay

        params = []
        for name, child in model.named_children():
            if name == 'transformer':
                lr = transformer_learning_rate if transformer_learning_rate is not None else learning_rate
                params.append({'params': child.parameters(), 'lr': lr})
            else:
                params.append({'params': child.parameters(), 'lr': learning_rate})

        super().__init__(params, **options)


class Adam(Adam):
    @ex.capture()
    def __init__(self, learning_rate, weight_decay):
        options = {
            'lr': learning_rate,
            'weight_decay': weight_decay
        }
        super().__init__(**options)