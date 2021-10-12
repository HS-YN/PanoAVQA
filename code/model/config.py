
from transformers.configuration_utils import PretrainedConfig

from exp import ex


'''
Refrain from directly modifying this configuration!
Please use /code/config.py for such purpose
'''
class ModelConfig(PretrainedConfig):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)

        self.pretrain_task_list = ['mask_lm', 'visual_feat', 'visual_coord', 'visual_label',
                                   'audio_feat', 'audio_coord', 'audio_label', 'vl_match',
                                   'al_match', 'qa', 'ground',
                                   'visual_vilbert', 'audio_vilbert']

        self.pretrain_types = []
        for task in self.pretrain_task_list:
            if hasattr(self, task) and getattr(self, task):
                self.pretrain_types.append(task)

        self.finetune_types = ['qa']
        if hasattr(self, 'use_grounding') and getattr(self, 'use_grounding'):
            self.finetune_types.append('grounding')

        self.audio_encoder = 'stereo' if self.use_stereo_audio else 'mono'
        if self.audio_coord_dim == 3:
            self.audio_encoder += '_st'
        elif self.audio_coord_dim == 2:
            self.audio_encoder += '_t'
        elif self.audio_coord_dim == 1:
            self.audio_encoder += '_s'

        if self.geometry in ['quaternion', 'spherical']:
            self.visual_coord_dim = 6
        elif self.geometry in ['angular', 'cartesian']:
            self.visual_coord_dim = 5
        else:
            self.visual_coord_dim = 0


        self.pretrain_loss_config = {
            'mask_lm': ((-1, self.vocab_size), 'ce', (-1,), 1),
            'visual_feat': ((-1, self.visual_feat_dim), 'l2', (-1, self.visual_feat_dim), self.loss_normalizer),
            'visual_coord': ((-1, self.visual_coord_dim), 'l2', (-1, self.visual_coord_dim), self.loss_normalizer),
            'visual_label': ((-1, self.visual_label_dim), 'ce_no_reduction', (-1,), self.loss_normalizer),
            'audio_feat': ((-1, 2, self.audio_feat_dim) if self.use_stereo_audio else (-1, self.audio_feat_dim), 
                           'l2', 
                           (-1, 2, self.audio_feat_dim) if self.use_stereo_audio else (-1, self.audio_feat_dim), 
                           self.loss_normalizer),
            'audio_harmonics': ((-1, 1), 'l2', (-1, 1), self.loss_normalizer),
            'audio_label': ((-1, self.audio_label_dim), 'ce_no_reduction', (-1,), self.loss_normalizer),
            'audio_coord': ((-1, self.audio_coord_dim), 'l2', (-1, self.audio_coord_dim), self.loss_normalizer),
            'vl_match': ((-1, 2), 'ce', (-1,), 1),
            'al_match': ((-1, 2), 'ce', (-1,), 1),
            'qa': ((-1, self.num_answers), 'ce_no_reduction', (-1,), 1),
            'ground': ((-1, max(0, self.visual_coord_dim-1)), 'l2_reduction', (-1, max(0, self.visual_coord_dim-1)), self.lambda_ground),
            'visual_vilbert': ((-1, self.visual_label_dim), 'kl', (-1,), self.loss_normalizer),
            'audio_vilbert': ((-1, self.audio_label_dim), 'kl', (-1,), self.loss_normalizer)
        }
