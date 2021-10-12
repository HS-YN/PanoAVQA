import json
from functools import partial

import torch
import numpy as np
from tqdm import tqdm

from exp import ex
from ckpt import load_ckpt
from metrics.logger import write_logs
from optimizer import get_optimizer
from metrics.logger import write_logs
from common import prepare_batch, get_all


grounding_error = torch.nn.MSELoss(reduction='none')


def get_accs(gt, prop, qtype):
    retval = {}
    correct = (gt == prop).float()
    retval['acc_total'] = [k.item() for k in correct]
    is_av = torch.Tensor(np.array(qtype) == 'a').float()
    retval['acc_av'] = [k.item() for i, k in enumerate(correct) if is_av[i] == 1]
    retval['acc_sp'] = [k.item() for i, k in enumerate(correct) if is_av[i] == 0]

    return retval


def get_errors(gt, prop, qtype):
    retval = {}
    errors = grounding_error(prop, gt).sum(1)
    retval['mse_total'] = [k.item() for k in errors]
    is_av = torch.Tensor(np.array(qtype) == 'a').float()
    retval['mse_av'] = [k.item() for i, k in enumerate(errors) if is_av[i] == 1]
    retval['mse_sp'] = [k.item() for i, k in enumerate(errors) if is_av[i] == 0]

    return retval



@ex.capture()
def _eval(log_path, ckpt_path, config_dir, max_epochs, pretrain_epochs, answer_path, learning_rate, ckpt_file,
           pretrain_learning_rate, pretrain_types, split_train, model_config, _config):
    dataloaders, _, tokenizer, model, criterion = get_all(data_modes=['test'])

    answer_dict = json.load(open(answer_path, 'r'))

    # print(model)
    # PRETRAIN
    model.load_state_dict(torch.load(ckpt_file)['model'])
    model.eval()
    qas = {}

    for _batch in tqdm(dataloaders['test'], total=len(dataloaders['test']), desc="Test"):
        batch, label, meta = prepare_batch(_batch)

        with torch.no_grad():
            stats = model(batch, label, ['qa', 'ground'])
            label['qa'] = label['qa'].cpu()
            label['ground'] = label['ground'].cpu()
            if 'ground_pred' in stats.keys():
                for i in range(len(label['ground'])):
                    qas[meta['question_id'][i]] = {
                        "video_id": meta["video_id"][i],
                        "question": meta["question"][i],
                        "ans_gt": answer_dict[label['qa'][i].item()],
                        "ans_pr": answer_dict[stats['answer_pred'][i].item()],
                        "grnd_gt": label['ground'][i].numpy().tolist(),
                        "grnd_pr": stats['ground_pred'][i].numpy().tolist(),
                        "grnd_err": grounding_error(label['ground'][i], stats['ground_pred'][i]).sum().item()
                    }
            else:
                assert False, "No grounding available"

    json.dump(qas, open('./{}.json'.format(ckpt_file.split('/')[-2]),'w'), indent=2)
