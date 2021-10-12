import torch
import numpy as np
from torch import nn
from transformers.activations import ACT2FN, gelu

from .decorator import full_model
from .pretrained import BertPreTrainedModel
from .input import *
from .attention import BertAttention, BertCrossAttention, \
                       BertIntermediate, BertOutput, BertLayer
from .pooler import LanguageHead, MatchHead, VisualHead, GroundHead, \
                    AudioHead, AnswerHead, BertPooler, poolerLoss, DummyHead

from exp import ex

'''
As we follow the naming convention of the official huggingface transformers repo,
there are a few nn modules with different naming convention.
THis dictionary tracks the class naming difference between lxmert and ours.
'''
pretrain_state_dict_mapper = {
    # 'huggingface_convention': 'lxmert_convention'
    'BertSelfAttention': 'BertAttention',
    'BertSelfOutput': 'BertAttOutput',
    'BertAttention': 'BertSelfattLayer',
    'BertCrossAttention': 'BertCrossattLayer',
    'LxmertCrossLayer': 'LXRTXLayer'
}


class LavitCrossLayer(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.la_attention = BertCrossAttention(model_config)
        self.av_attention = BertCrossAttention(model_config)
        self.vl_attention = BertCrossAttention(model_config)

        self.l_self_att = BertAttention(model_config)
        self.v_self_att = BertAttention(model_config)
        self.a_self_att = BertAttention(model_config)

        self.l_inter = BertIntermediate(model_config)
        self.l_output = BertOutput(model_config)
        self.v_inter = BertIntermediate(model_config)
        self.v_output = BertOutput(model_config)
        self.a_inter = BertIntermediate(model_config)
        self.a_output = BertOutput(model_config)

    def cross_att(self, l_feat, l_mask, a_feat, a_mask, v_feat, v_mask):
        l_out = self.la_attention(l_feat, a_feat, a_mask)
        a_out = self.av_attention(a_feat, v_feat, v_mask)
        v_out = self.vl_attention(v_feat, l_feat, l_mask)
        return l_out, a_out, v_out

    def self_att(self, l_feat, l_mask, a_feat, a_mask, v_feat, v_mask):
        l_feat = self.l_self_att(l_feat, l_mask)
        a_feat = self.a_self_att(a_feat, a_mask)
        v_feat = self.v_self_att(v_feat, v_mask)

        l_out = self.l_output(self.l_inter(l_feat), l_feat)
        a_out = self.a_output(self.a_inter(a_feat), a_feat)
        v_out = self.v_output(self.v_inter(v_feat), v_feat)
        return l_out, a_out, v_out

    def forward(self, l_feat, l_mask, a_feat, a_mask, v_feat, v_mask):
        l_out, a_out, v_out = self.cross_att(l_feat, l_mask, a_feat, a_mask, v_feat, v_mask)
        l_out, a_out, v_out = self.self_att(l_out, l_mask, a_out, a_mask, v_out, v_mask)
        return l_out, a_out, v_out


class LavitEncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        if not model_config.no_coord:
            self.v_fc = VisualFeatEncoder(model_config)
        else:
            self.v_fc = VisualFeatNoCoordEncoder(model_config)

        if model_config.audio_encoder == 'mono':
            self.a_fc = AudioMonoEncoder(model_config)
        elif model_config.audio_encoder == 'mono_s':
            self.a_fc = AudioMonoSEncoder(model_config)
        elif model_config.audio_encoder == 'mono_t':
            self.a_fc = AudioMonoTEncoder(model_config)
        elif model_config.audio_encoder == 'mono_st':
            self.a_fc = AudioMonoSTEncoder(model_config)
        elif model_config.audio_encoder == 'stereo':
            self.a_fc = AudioStereoEncoder(model_config)
        elif model_config.audio_encoder == 'stereo_s':
            self.a_fc = AudioStereoSEncoder(model_config)
        elif model_config.audio_encoder == 'stereo_t':
            self.a_fc = AudioStereoTEncoder(model_config)
        elif model_config.audio_encoder == 'stereo_st':
            self.a_fc = AudioStereoSTEncoder(model_config)

        self.layer = nn.ModuleList(
            [BertLayer(model_config) for _ in range(model_config.l_layers)]
        )
        self.x_layers = nn.ModuleList(
            [LavitCrossLayer(model_config) for _ in range(model_config.x_layers)]
        )
        self.v_layers = nn.ModuleList(
            [BertLayer(model_config) for _ in range(model_config.v_layers)]
        )
        self.a_layers = nn.ModuleList(
            [BertLayer(model_config) for _ in range(model_config.a_layers)]
        )

    def forward(self, l_feat, l_mask, a_feat, a_mask=None, v_feat=None, v_mask=None):
        v_feat = self.v_fc(v_feat)
        a_feat = self.a_fc(a_feat)

        for layer_module in self.layer:
            l_feat = layer_module(l_feat, l_mask)
        for layer_module in self.v_layers:
            v_feat = layer_module(v_feat, v_mask)
        for layer_module in self.a_layers:
            a_feat = layer_module(a_feat, a_mask)
        for layer_module in self.x_layers:
            l_feat, a_feat, v_feat = layer_module(l_feat, l_mask, a_feat, a_mask, v_feat, v_mask)

        return l_feat, a_feat, v_feat   
        

class LavitModel(BertPreTrainedModel):
    def __init__(self, model_config, cache_path):
        super().__init__(model_config, cache_path)
        self.config = model_config

        self.embeddings = BertEmbeddings(model_config)
        self.encoder = LavitEncoder(model_config)
        self.pooler = BertPooler(model_config)
        self.a_pooler = BertPooler(model_config)
        self.v_pooler = BertPooler(model_config)
        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, l_mask=None, a_feat=None, a_mask=None,
                v_feat=None, v_mask=None):
        if l_mask is None:
            l_mask = torch.ones_like(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        e_l_mask = l_mask.unsqueeze(1).unsqueeze(2)
        e_l_mask = e_l_mask.to(dtype=next(self.parameters()).dtype)
        e_l_mask = (1.0 - e_l_mask) * -10000.0

        if v_mask is not None:
            e_v_mask = v_mask.unsqueeze(1).unsqueeze(2)
            e_v_mask = e_v_mask.to(dtype=next(self.parameters()).dtype)
            e_v_mask = (1.0 - e_v_mask) * -10000.0
        else:
            e_v_mask = None

        if a_mask is not None:
            e_a_mask = a_mask.unsqueeze(1).unsqueeze(2)
            e_a_mask = e_a_mask.to(dtype=next(self.parameters()).dtype)
            e_a_mask = (1.0 - e_a_mask) * -10000.0
        else:
            e_a_mask = None

        l_feat = self.embeddings(input_ids, token_type_ids)
        l_feat, a_feat, v_feat = self.encoder(l_feat, e_l_mask, a_feat, e_a_mask, v_feat, e_v_mask)
        l_pool = self.pooler(l_feat)
        a_pool = self.a_pooler(a_feat)
        v_pool = self.v_pooler(v_feat)

        return (l_feat, a_feat, v_feat), (l_pool, a_pool, v_pool)


class LavitPretraining(BertPreTrainedModel):
    def __init__(self, model_config, cache_path):
        super().__init__(model_config, cache_path)
        self.config = model_config
        self.num_modality = 1

        self.bert = LavitModel(model_config, cache_path)

        # mask_lm
        self.cls = LanguageHead(model_config, bert_weights=self.bert.embeddings.word_embeddings.weight)

        if 'visual' in [x[:6] for x in model_config.pretrain_types]:
            self.v_head = VisualHead(model_config)
            self.num_modality += 1
        else:
            self.v_head = DummyHead()

        if 'audio' in [x[:5] for x in model_config.pretrain_types]:
            self.a_head = AudioHead(model_config)
            self.num_modality += 1
        else:
            self.a_head = DummyHead()

        if 'vl_match' in model_config.pretrain_types:
            self.vl_head = MatchHead(model_config)
        else:
            self.vl_head = DummyHead()

        if 'al_match' in model_config.pretrain_types:
            self.al_head = MatchHead(model_config)
        else:
            self.al_head = DummyHead()

        self.answer_head = AnswerHead(model_config, num_modality=self.num_modality)

        if 'ground' in model_config.pretrain_types:
            self.ground = GroundHead(model_config,  num_modality=self.num_modality)
        else:
            self.ground = DummyHead()

        self.apply(self.init_bert_weights)

    def forward(self, input_ids, token_type_ids=None, l_mask=None, a_feat=None, a_mask=None,
                v_feat=None, v_mask=None):
        (l_out, a_out, v_out), (l_head, a_head, v_head) = self.bert(
            input_ids, token_type_ids, l_mask, 
            a_feat=a_feat, a_mask=None, v_feat=v_feat, v_mask=None
        )

        pred = {}
        pred['mask_lm'] = self.cls(l_out)
        pred.update(self.a_head(a_out))
        pred.update(self.v_head(v_out))
        pred['vl_match'] = self.vl_head([v_head, l_head])
        pred['al_match'] = self.al_head([a_head, l_head])
        pred['qa'] = self.answer_head([l_head, a_head, v_head])
        pred['ground'] = self.ground([l_head, a_head, v_head])

        return pred


@full_model
class Lavit(nn.Module):
    def __init__(self, model_config, cache_path):
        super().__init__()
        self.config = model_config
        self.pretrain_loss_config = model_config.pretrain_loss_config
        self.audio_encoder = model_config.audio_encoder
        self.model = LavitPretraining.from_pretrained(model_config, cache_path)

    def forward(self, batch, label, tasks):
        input_ids = batch.get('l_feat')
        token_type_ids = None
        l_mask = batch.get('l_mask')
        v_feat = (batch.get('v_feat'), batch.get('v_coord'))
        v_mask = batch.get('v_mask')
        a_feat = (batch.get('a_feat'), batch.get('a_coord'))
        a_mask = batch.get('a_mask')

        pred = self.model(input_ids, token_type_ids, l_mask,
                          a_feat=a_feat, a_mask=None, v_feat=v_feat, v_mask=None)

        mask = {
            "visual_feat": batch.get('v_mask'),
            "visual_label": label.get('visual_score'),
            "visual_coord": batch.get('v_mask'),
            "audio_feat": batch.get('a_mask'),
            "audio_label": label.get("audio_score"),
            "audio_coord": batch.get("a_mask"),
            "qa": label.get('qa_valid')
        }
        if mask['visual_label'] is not None:
            mask['visual_label'] *= mask['visual_feat']
        if mask["audio_label"] is not None:
            mask['audio_label'] *= mask['audio_feat']

        if 'audio_coord' in label.keys() and self.config.audio_coord_dim != label['audio_coord'].shape[-1]:
            if self.config.audio_coord_dim == 1:
                label['audio_coord'] = label['audio_coord'][:, :, -1]
            elif self.config.audio_coord_dim == 2:
                label['audio_coord'] = label['audio_coord'][:, :, :-1]

        total_loss = 0.
        loss = {}
        for task in tasks:
            output_shape, loss_type, label_shape, weight = self.pretrain_loss_config[task]
            loss_func = poolerLoss[loss_type]

            if task in label.keys() and label[task] is not None:
                task_loss = loss_func(
                    pred[task].view(*output_shape),
                    label[task].view(*label_shape)
                )

                if task_loss.dim() > 1:
                    task_loss = task_loss.mean(1)
                    if task_loss.dim() > 1:
                        task_loss = task_loss.mean(1)
                    task_loss = (task_loss * mask[task].view(-1)).mean()
                elif task_loss.dim() == 1:
                    task_loss = (task_loss * mask[task].view(-1)).mean()

                task_loss = task_loss * weight
                total_loss += task_loss
                loss[f'loss_{task}'] = task_loss.detach()
        

        answer_pred = np.argmax(pred['qa'].detach().cpu(), 1) if 'qa' in pred.keys() else None
        ground_pred = pred['ground'].detach().cpu() if 'ground' in pred.keys() and pred['ground'] is not None else None

        return {'total_loss': total_loss, 
                'answer_pred': answer_pred, 
                'ground_pred': ground_pred, 
                **loss}
