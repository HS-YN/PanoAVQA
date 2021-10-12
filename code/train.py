import json
from functools import partial

import torch
import numpy as np
from tqdm import tqdm

from exp import ex
from ckpt import save_ckpt
from metrics.logger import write_logs
from optimizer import get_optimizer
from metrics.logger import write_logs
from common import prepare_batch, get_all


grounding_error = torch.nn.MSELoss(reduction='none')


@ex.capture()
def get_pretrain_task(split, epoch, pretrain_epochs, model_name, pretrain_types, model_config):
    if split != 'pretrain':
        return model_config.finetune_types
    else:
        if model_name in ["bert", "bert_scratch"]:
            return ["mask_lm", "ground"]

        elif model_name == 'lxmert':
            if epoch < (pretrain_epochs/2):
                return ["mask_lm", "vl_match", "visual_feat", "visual_label"]
            else:
                return ["mask_lm", "vl_match", "visual_feat", "visual_label", "qa"]
        else: #lavit
            return pretrain_types


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
def _train(log_path, ckpt_path, config_dir, max_epochs, pretrain_epochs, answer_path, learning_rate,
           pretrain_learning_rate, pretrain_types, split_train, model_config, _config):
    dataloaders, _, tokenizer, model, criterion = get_all(data_modes=['pretrain', 'train', 'preval', 'val', 'test'])
    print("[LOG] Logging to {}".format(log_path / config_dir))
    logger = torch.utils.tensorboard.SummaryWriter(log_path / config_dir)

    answer_dict = json.load(open(answer_path, 'r'))

    # print(model)
    # PRETRAIN
    it = 0
    if pretrain_epochs > 0 and hasattr(model, 'run_pretrain'):
        if split_train:
            model.fix_transformer(True)

        optimizer, scheduler = get_optimizer(model,
                                             dataloaders['pretrain'].dataset.t_total,
                                             learning_rate=pretrain_learning_rate)
        optimizer.zero_grad()


        for epoch in range(pretrain_epochs):
            
            model.train()
            
            pretrain_tasks = get_pretrain_task('pretrain', epoch)
            print(pretrain_tasks)
            
            for _batch in tqdm(dataloaders['pretrain'], total=len(dataloaders['pretrain']), desc=f"Pretrain e{epoch}"):
                batch, label, meta = prepare_batch(_batch)

                stats = model(batch, label, pretrain_tasks)
                stats['total_loss'].backward()
                # if debug:
                #     with torch.autograd.set_detect_anomaly(True):
                #         stats = model(batch, label)

                scheduler.accumulated += 1
                if scheduler.accumulated >= scheduler.grad_acc_steps:
                    optimizer.step()
                    scheduler.step()
                    scheduler.accumulated = 0
                    optimizer.zero_grad()

                if it % 20 == 0:
                    write_logs(logger, it, optimizer.param_groups[0]['lr'], stats, meta, 'train')

                it += 1

            # Evaluate
            model.eval()
            scheduler.accumulated = 0
            eval_stats = {}
            acc = {'acc_total': [], 'acc_sp': [], 'acc_av': []}
            mse = {'mse_total': [], 'mse_sp': [], 'mse_av': []}
            qas = []
            for _batch in tqdm(dataloaders['preval'], total=len(dataloaders['preval']), desc="Valid"):
                batch, label, meta = prepare_batch(_batch)

                with torch.no_grad():
                    stats = model(batch, label, pretrain_tasks)
                    for k, v in get_accs(label['qa'].cpu(), stats['answer_pred'], meta['question_type']).items():
                        acc[k].extend(v)
                    if 'ground_pred' in stats.keys():
                        for k, v in get_errors(label['ground'].cpu(), stats['ground_pred'], meta['question_type']).items():
                            mse[k].extend(v)
                        qas.append("[{}] {} / (GT) {} (PROP) {} / (GT) {} (PROP) {}".format(
                            meta['video_id'][0], meta['question'][0], answer_dict[label['qa'][0].item()], answer_dict[stats['answer_pred'][0].item()],
                            str([f"{i:.3f}" for i in label['ground'][0]]), str([f"{i:.3f}" for i in stats['ground_pred'][0]])
                            ))
                    else:
                        qas.append("[{}] {} / (GT) {} (PROP) {}".format(
                            meta['video_id'][0], meta['question'][0], answer_dict[label['qa'][0].item()], answer_dict[stats['answer_pred'][0].item()]
                            ))

                for k, v in stats.items():
                    if type(v) == torch.Tensor and v.dim() == 0:
                        eval_stats[k] = v.item() if k not in eval_stats.keys() else eval_stats[k] + v.item()
            for k, v in acc.items():
                eval_stats[k] = sum(v) / len(v)
            if 'ground_pred' in stats.keys():
                for k, v in mse.items():
                    eval_stats[k] = sum(v) / len(v)

            print([f"{k}: {v:.4f}" for k,v in eval_stats.items()])
            eval_stats['example'] = '\n\n'.join(qas)
            write_logs(logger, epoch, None, eval_stats, meta, 'eval')
            save_ckpt(epoch, stats['total_loss'].item(), model)
            
    # TRAIN
    if split_train:
        model.fix_transformer(False)
    optimizer, scheduler = get_optimizer(model, dataloaders['train'].dataset.t_total, learning_rate=learning_rate)
    optimizer.zero_grad()

    for epoch in range(pretrain_epochs, max_epochs):

        train_tasks = get_pretrain_task('train', epoch)
        print(train_tasks)

        model.train()
        for _batch in tqdm(dataloaders['train'], total=len(dataloaders['train']), desc=f"Train e{epoch}"):
            batch, label, meta = prepare_batch(_batch)

            stats = model(batch, label, train_tasks)
            stats['total_loss'].backward()

            scheduler.accumulated += 1
            if scheduler.accumulated >= scheduler.grad_acc_steps:
                optimizer.step()
                scheduler.step()
                scheduler.accumulated = 0
                optimizer.zero_grad()

            if it % 20 == 0:
                write_logs(logger, it, optimizer.param_groups[0]['lr'], stats, meta, 'train')
            it += 1

        # Evaluate
        model.eval()
        scheduler.accumulated = 0
        eval_stats = {}
        acc = {'acc_total': [], 'acc_sp': [], 'acc_av': []}
        mse = {'mse_total': [], 'mse_sp': [], 'mse_av': []}
        qas = []
        for _batch in tqdm(dataloaders['val'], total=len(dataloaders['val']), desc="Valid"):
            batch, label, meta = prepare_batch(_batch)

            with torch.no_grad():
                stats = model(batch, label, train_tasks)
                for k, v in get_accs(label['qa'].cpu(), stats['answer_pred'], meta['question_type']).items():
                    acc[k].extend(v)
                if 'ground_pred' in stats.keys():
                    for k, v in get_errors(label['ground'].cpu(), stats['ground_pred'], meta['question_type']).items():
                        mse[k].extend(v)
                    qas.append("[{}] {} / (GT) {} (PROP) {} / (GT) {} (PROP) {}".format(
                        meta['video_id'][0], meta['question'][0], answer_dict[label['qa'][0].item()], answer_dict[stats['answer_pred'][0].item()],
                        str([f"{i:.3f}" for i in label['ground'][0]]), str([f"{i:.3f}" for i in stats['ground_pred'][0]])
                        ))
                else:
                    qas.append("[{}] {} / (GT) {} (PROP) {}".format(
                        meta['video_id'][0], meta['question'][0], answer_dict[label['qa'][0].item()], answer_dict[stats['answer_pred'][0].item()]
                        ))

            for k, v in stats.items():
                if type(v) == torch.Tensor and v.dim() == 0:
                    eval_stats[k] = v.item() if k not in eval_stats.keys() else eval_stats[k] + v.item()

        for k, v in acc.items():
            eval_stats[k] = sum(v) / len(v)
        if 'ground_pred' in stats.keys():
            for k, v in mse.items():
                eval_stats[k] = sum(v) / len(v)

        print([f"{k}: {v:.4f}" for k,v in eval_stats.items()])
        eval_stats['example'] = '\n\n'.join(qas)
        write_logs(logger, epoch, None, eval_stats, meta, 'eval')
        save_ckpt(epoch, stats['total_loss'].item(), model)

    # Test
    model.eval()
    scheduler.accumulated = 0
    eval_stats = {}
    acc = {'acc_total': [], 'acc_sp': [], 'acc_av': []}
    mse = {'mse_total': [], 'mse_sp': [], 'mse_av': []}
    qas = []
    for _batch in tqdm(dataloaders['test'], total=len(dataloaders['test']), desc="Test"):
        batch, label, meta = prepare_batch(_batch)

        with torch.no_grad():
            stats = model(batch, label, train_tasks)
            for k, v in get_accs(label['qa'].cpu(), stats['answer_pred'], meta['question_type']).items():
                acc[k].extend(v)
            if 'ground_pred' in stats.keys():
                for k, v in get_errors(label['ground'].cpu(), stats['ground_pred'], meta['question_type']).items():
                    mse[k].extend(v)
                qas.extend(["[{}] {} / (GT) {} (PROP) {} / (GT) {} (PROP) {}".format(
                    meta['video_id'][i], meta['question'][i], answer_dict[label['qa'][i].item()], answer_dict[stats['answer_pred'][i].item()],
                    str([f"{i:.3f}" for i in label['ground'][i]]), str([f"{i:.3f}" for i in stats['ground_pred'][i]])
                    ) for i in range(len(_batch))])
            else:
                qas.extend(["[{}] {} / (GT) {} (PROP) {}".format(
                    meta['video_id'][i], meta['question'][i], answer_dict[label['qa'][i].item()], answer_dict[stats['answer_pred'][i].item()]
                    ) for i in range(len(_batch))])

        for k, v in stats.items():
            if type(v) == torch.Tensor and v.dim() == 0:
                eval_stats[k] = v.item() if k not in eval_stats.keys() else eval_stats[k] + v.item()

    for k, v in acc.items():
        eval_stats[k] = sum(v) / len(v)
    if 'ground_pred' in stats.keys():
        for k, v in mse.items():
            eval_stats[k] = sum(v) / len(v)

    print([f"{k}: {v:.4f}" for k,v in eval_stats.items()])
    eval_stats['example'] = '\n\n'.join(qas)
    write_logs(logger, 0, None, eval_stats, meta, 'test')
    print("GPU Allocation: ", torch.cuda.memory_stats()["allocation.all.peak"], "MB")
