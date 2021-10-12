import math

import torch
import torch.nn as nn
import torch.nn.functional as F

poolerLoss = {
    'ce': nn.CrossEntropyLoss(ignore_index=-1),
    'l2': nn.SmoothL1Loss(reduction='none'),
    'l2_reduction': nn.SmoothL1Loss(),
    'ce_no_reduction': nn.CrossEntropyLoss(ignore_index=-1, reduction='none'),
    'kl': lambda x,y: nn.KLDivLoss(reduction='none')(F.log_softmax(x, dim=2), y)
}

def gelu(x):
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

ACT2FN = {"gelu": gelu, "relu": nn.functional.relu}


class GeLU(nn.Module):
    def __init__(self):
        # torch.nn.functional.gelu is not a Module subclass
        super().__init__()

    def forward(self, x):
        return gelu(x)


class BertPooler(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.dense = nn.Linear(model_config.hidden_size, model_config.hidden_size)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        cls_tensor = hidden_states[:, 0]
        output = self.activation(self.dense(cls_tensor))
        return output


class BertPredictionHeadTransform(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.dense = nn.Linear(model_config.hidden_size, model_config.hidden_size)
        if isinstance(model_config.hidden_act, str):
            self.transform_act_fn = ACT2FN[model_config.hidden_act]
        else:
            self.transform_act_fn = model_config.hidden_act
        self.LayerNorm = nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)

    def forward(self, hidden_states):
        hidden_states = self.dense(hidden_states)
        hidden_states = self.transform_act_fn(hidden_states)
        hidden_states = self.LayerNorm(hidden_states)
        return hidden_states


class BertLMPredictionHead(nn.Module):
    def __init__(self, model_config, bert_weights=None):
        super().__init__()
        self.transform = BertPredictionHeadTransform(model_config)

        if bert_weights is None:
            self.decoder = nn.Linear(model_config.hidden_size, model_config.vocab_size, bias=False)
            self.bias = nn.Parameter(torch.zeros(model_config.vocab_size))
        else:
            self.decoder = nn.Linear(bert_weights.size(1), bert_weights.size(0), bias=False)
            self.decoder.weight = bert_weights
            self.bias = nn.Parameter(torch.zeros(bert_weights.size(0)))
        
    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = self.decoder(hidden_states) + self.bias
        return output


class LanguageHead(nn.Module):
    def __init__(self, model_config, bert_weights=None):
        super().__init__()
        self.predictions = BertLMPredictionHead(model_config, bert_weights)

    def forward(self, hidden_states):
        return self.predictions(hidden_states)


class VisualHead(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(model_config)
        self.tasks = {}

        if 'visual_feat' in model_config.pretrain_types:
            self.feat_decoder = nn.Linear(model_config.hidden_size, model_config.visual_feat_dim)
            self.tasks["visual_feat"] = self.feat_decoder
        if 'visual_coord' in model_config.pretrain_types:
            self.coord_decoder = nn.Linear(model_config.hidden_size, model_config.visual_coord_dim)
            self.tasks["visual_coord"] = self.coord_decoder
        if 'visual_label' in model_config.pretrain_types:
            self.label_decoder = nn.Linear(model_config.hidden_size, model_config.visual_label_dim)
            self.tasks["visual_label"] = self.label_decoder

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for task, decoder in self.tasks.items():
            output[task] = decoder(hidden_states)
        return output


class AudioHead(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        self.transform = BertPredictionHeadTransform(model_config)
        self.tasks = {}
        self.feat_dim = model_config.audio_feat_dim * 2 if model_config.use_stereo_audio else model_config.audio_feat_dim

        if 'audio_feat' in model_config.pretrain_types:
            self.feat_decoder = nn.Linear(model_config.hidden_size, self.feat_dim)
            self.tasks["audio_feat"] = self.feat_decoder
        if 'audio_harmonics' in model_config.pretrain_types:
            self.harm_decoder = nn.Linear(model_config.hidden_size, 1)
            self.tasks["audio_harmonics"] = self.harm_decoder # regression
        if 'audio_harmonics_reg' in model_config.pretrain_types:
            self.harm_decoder = nn.Linear(model_config.hidden_size, 1)
            self.tasks["audio_harmonics_reg"] = self.harm_decoder
        if 'audio_harmonics_bin' in model_config.pretrain_types:
            self.harm_decoder = nn.Linear(model_config.hidden_size, 3)  # -1, 0, 1
            self.tasks["audio_harmonics_reg"] = self.harm_decoder
        if 'audio_label' in model_config.pretrain_types:
            self.label_decoder = nn.Linear(model_config.hidden_size, model_config.audio_label_dim)
            self.tasks["audio_label"] = self.label_decoder
        if 'audio_coord' in model_config.pretrain_types:
            self.coord_decoder = nn.Linear(model_config.hidden_size, model_config.audio_coord_dim)
            self.tasks["audio_coord"] = self.coord_decoder

    def forward(self, hidden_states):
        hidden_states = self.transform(hidden_states)
        output = {}
        for task, decoder in self.tasks.items():
            output[task] = decoder(hidden_states).squeeze()
        return output


class AnswerHead(nn.Module):
    def __init__(self, model_config, num_modality=1):
        super().__init__()
        in_dim = model_config.hidden_size
        hid_dim = 2 * in_dim
        if model_config.use_concat_decoder and num_modality > 1:
            in_dim = num_modality * in_dim
            hid_dim = 2 * in_dim
        num_answers = model_config.num_answers

        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            GeLU(),
            nn.LayerNorm(hid_dim, eps=model_config.layer_norm_eps),
            nn.Linear(hid_dim, num_answers)
        )

    def forward(self, x):
        if type(x) == list:
            return self.logit_fc(torch.cat(x, 1))
        else:
            return self.logit_fc(x)


class GroundHead(nn.Module):
    def __init__(self, model_config, num_modality=1):
        super().__init__()
        in_dim = model_config.hidden_size
        hid_dim = 2 * in_dim
        if model_config.use_concat_decoder and num_modality > 1:
            in_dim = num_modality * in_dim
            hid_dim = 2 * in_dim
        ground_dim = model_config.visual_coord_dim - 1

        self.logit_fc = nn.Sequential(
            nn.Linear(in_dim, hid_dim),
            GeLU(),
            nn.LayerNorm(hid_dim, eps=model_config.layer_norm_eps),
            nn.Linear(hid_dim, ground_dim)
        )

    def forward(self, x):
        if type(x) == list:
            return self.logit_fc(torch.cat(x, 1))
        else:
            return self.logit_fc(x)


class MatchHead(nn.Module):
    def __init__(self, model_config, use_concat_decoder=None):
        super().__init__()
        self.use_concat_decoder = use_concat_decoder if use_concat_decoder is not None else model_config.use_concat_decoder
        if self.use_concat_decoder:
            self.match_head = nn.Linear(2 * model_config.hidden_size, 2)
        else:
            self.match_head = nn.Linear(model_config.hidden_size, 2)

    def forward(self, x):
        if type(x) == list:
            return self.match_head(torch.cat(x, 1))
        else:
            return self.match_head(x)


class DummyHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return None