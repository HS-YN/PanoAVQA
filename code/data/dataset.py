import re
import json
from itertools import chain

import torch
import numpy as np
import random
from munch import Munch

from exp import ex
from .load import load
from utils import merge_dict, one_hot_vectorize
#from .utils import pad


audio_pad_dict = {
    'orig': 18,
    'pool_2': 9,
    'pool_4': 6,
    'pool_all': 1,
    'top_3': 18,
    'top_5': 18
}

def get_dataset(modes=[]):
    data, video, tokenizer = load(modes)
    outputs = {}
    for mode in sorted(list(data.keys())):
        print(f"[LOG] Loading {mode} split... ", end='')
        mode_feat = {}
        mode_ids = set([x['video_id'] for x in data[mode].values()])
        for modality, feature in video.items():
            mode_feat[modality] = {k: v for k, v in feature.items() if k in mode_ids}
        print("({} video features)".format(len(mode_ids)))
        outputs[mode] = Dataset(data=data[mode], mode=mode, video=mode_feat, tokenizer=tokenizer)
    return outputs, video, tokenizer


class Dataset(torch.utils.data.Dataset):
    @ex.capture()
    def __init__(self, data, mode, video, tokenizer, model_config, device, feature_mask_rate,
                 answer_path, num_objects):
        self.data = data
        self.video = video
        self.ids = list(self.data.keys())
        self.tokenizer = tokenizer
        self.device = device
        self.mode = mode # pretrain, train, val, test
        self.feature_mask_rate = feature_mask_rate
        self.model_config = Munch(model_config)
        self.pretrain_types = self.model_config.pretrain_types if (self.mode == "pretrain" or self.mode == "preval") else ['qa']

        self.answer_label = json.load(open(answer_path, 'r'))

        self.use_cls_token = self.model_config.use_cls_token
        self.num_objects = num_objects
        self.audio_objects = audio_pad_dict[self.model_config.audio_feature]
        self.geometry = self.model_config.geometry

        self.feature_names = []
        if self.model_config.visual_feature is not None:
            self.feature_names.append('visual')
        if self.model_config.audio_feature is not None:
            self.feature_names.append('audio')

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        qid = self.ids[idx]
        datum = self.data[qid].copy()
        id = datum['video_id']

        # apply masking
        # check the number of features with norm > 0
        # probabilistically mask and generate masking
        if 'visual' in self.feature_names:
            video_feat = self.video['visual'][id]

            datum['v_feat'] = video_feat['embedding']
            datum['v_coord'] = self.video['coordinate'][id]

        if 'audio' in self.feature_names:
            audio_feat = self.video['audio'][id]
            '''
            if self.model_config.use_stereo_audio:
                audio_feat_embedding = audio_feat['embedding'][1:, :, :]
            else:
                audio_feat_embedding = audio_feat['embedding'][0, :, :]
            '''
            datum['a_feat'] = audio_feat['embedding']
            datum['a_coord'] = audio_feat['coordinate']

        return datum

    def prepare_pretrain(self, datum, batch_question_tokens):

        label = {}
        metadata = {}

        id = datum['video_id']
        video_feat = datum['v_feat']
        video_coord = datum['v_coord']
        video_class = self.video['visual'][id]['classes']
        video_score = self.video['visual'][id]['score']

        if 'visual' in self.feature_names:
            # Select random visual feature in batch
            rand_id = self.data[random.choice(self.ids)]['video_id']
            while rand_id == id:
                rand_id = self.data[random.choice(self.ids)]['video_id']
            rand_feat = self.video['visual'][rand_id]['embedding']
            if rand_feat.shape[0] == 0:
                rand_feat = np.zeros_like(video_feat)

            if 'vl_match' in self.pretrain_types:
                if random.random() > 0.5:
                    video_feat, rand_feat = rand_feat, video_feat
                    video_coord = self.video['visual'][rand_id]['coordinate']
                    video_class = self.video['visual'][rand_id]['classes']
                    video_score = self.video['visual'][rand_id]['score']
                    label['vl_match'] = 1
                else:
                    label['vl_match'] = 0

            # CLS_v token
            if self.use_cls_token:
                video_cls = np.expand_dims(np.mean(video_feat, axis=0), axis=0)
                coord_cls = np.expand_dims(np.mean(video_coord, axis=0), axis=0)
                mask_loop_range = (1, min(len(video_feat) + 1, self.num_objects))
                assert video_cls.shape == (1, 2048)

                video_feat = np.concatenate([video_cls, video_feat], axis=0)
                video_coord = np.concatenate([coord_cls, video_coord], axis=0)
                video_class = np.concatenate([[0], video_class])
                video_score = np.concatenate([[0], video_score])
            else:
                mask_loop_range = (0, min(len(video_feat), self.num_objects))

            # zero-pad
            video_feat = video_feat[:self.num_objects]
            video_coord = video_coord[:self.num_objects]
            video_class = video_class[:self.num_objects]
            video_score = video_score[:self.num_objects]
            if video_feat.shape[0] < self.num_objects:
                surplus = self.num_objects - video_feat.shape[0]
                video_feat = np.concatenate([video_feat, np.zeros((surplus, video_feat.shape[1]))], axis=0)
                video_coord = np.concatenate([video_coord, np.zeros((surplus, video_coord.shape[1]))], axis=0)
                video_class = np.concatenate([video_class, np.zeros(surplus)])
                video_score = np.concatenate([video_score, np.zeros(surplus)])

            # Apply masking for visual feature
            if 'visual' in [x[:6] for x in self.pretrain_types]:
                mask_feat = video_feat.copy()
                visual_mask = [0. for _ in range(len(mask_feat))]

                for i in range(*mask_loop_range):
                    prob = random.random()

                    if prob < self.feature_mask_rate:
                        prob /= self.feature_mask_rate

                        if prob < 0.8:
                            mask_feat[i, :] = 0.

                        elif prob < 0.9:
                            rand_idx = random.choice(range(rand_feat.shape[0]))
                            mask_feat[i, :] = rand_feat[rand_idx, :]
                        visual_mask[i] = 1.

                datum['v_feat'] = mask_feat
                datum['v_coord'] = video_coord
                datum['v_mask'] = visual_mask

                label['visual_feat'] = video_feat
                label['visual_coord'] = video_coord
                label['visual_label'] = video_class.astype(np.long)
                label['visual_score'] = video_score

            else:
                datum['v_feat'] = video_feat
                datum['v_coord'] = video_coord


        audio_feat = datum['a_feat'] # embedding
        audio_coord = datum['a_coord']
        audio_score = self.video['audio'][id]['score']
        audio_class = self.video['audio'][id]['classes']
        audio_harmo = self.video['audio'][id]['harmonics']

        if 'audio' in self.feature_names:
            rand_id = self.data[random.choice(self.ids)]['video_id']
            while rand_id == id:
                rand_id = self.data[random.choice(self.ids)]['video_id']

            if self.model_config.use_stereo_audio:
                audio_feat = audio_feat[1:, :, :]
                rand_feat = self.video['audio'][rand_id]['embedding'][1:, :, :]
                assert audio_feat.shape[0] == rand_feat.shape[0]
            else:
                audio_feat = np.expand_dims(audio_feat[0, :, :], axis=0)
                rand_feat = np.expand_dims(self.video['audio'][rand_id]['embedding'][0, :, :], axis=0)

            if 'al_match' in self.pretrain_types:
                if random.random() > 0.5:
                    audio_feat, rand_feat = rand_feat, audio_feat
                    audio_coord = self.video['audio'][rand_id]['coordinate']
                    audio_score = self.video['audio'][rand_id]['score']
                    audio_class = self.video['audio'][rand_id]['classes']
                    audio_harmo = self.video['audio'][rand_id]['harmonics']
                    label['al_match'] = 1
                else:
                    label['al_match'] = 0

            # CLS_v token
            if self.use_cls_token:
                #rint(">>>" , audio_feat.shape)
                audio_cls = np.expand_dims(np.mean(audio_feat, axis=1), axis=1)
                coord_cls = np.expand_dims(np.mean(audio_coord, axis=0), axis=0)
                mask_loop_range = (1, min(len(audio_feat[0]) + 1, self.audio_objects))

                audio_feat = np.concatenate([audio_cls, audio_feat], axis=1)
                audio_coord = np.concatenate([coord_cls, audio_coord], axis=0)
                audio_score = np.concatenate([[0], audio_score])
                audio_class = np.concatenate([[0], audio_class])
                audio_harmo = np.concatenate([[0], audio_harmo])
            else:
                mask_loop_range = (0, min(len(audio_feat[0]), self.audio_objects))

            # zero-pad
            audio_feat = audio_feat[:self.audio_objects]
            audio_coord = audio_coord[:self.audio_objects]
            audio_class = audio_class[:self.audio_objects]
            audio_score = audio_score[:self.audio_objects]
            audio_harmo = audio_harmo[:self.audio_objects]
            if audio_feat.shape[1] < self.audio_objects:
                surplus = self.audio_objects - audio_feat.shape[1]
                audio_feat = np.concatenate([audio_feat, np.zeros((audio_feat.shape[0], surplus, audio_feat.shape[2]))], axis=1)
                audio_coord = np.concatenate([audio_coord, np.zeros((surplus, audio_coord.shape[1]))], axis=0)
                audio_class = np.concatenate([audio_class, np.zeros(surplus)])
                audio_score = np.concatenate([audio_score, np.zeros(surplus)])
                audio_harmo = np.concatenate([audio_harmo, np.zeros(surplus)])


            # Do masking
            if 'audio' in [x[:5] for x in self.pretrain_types]:
                mask_feat = audio_feat.copy()
                audio_mask = [0. for _ in range(len(mask_feat[0]))]
                
                for i in range(*mask_loop_range):
                    prob = random.random()

                    if prob < self.feature_mask_rate:
                        prob /= self.feature_mask_rate

                        if prob < 0.8:
                            mask_feat[:, i, :] = 0.

                        elif prob < 0.9:
                            rand_idx = random.choice(range(rand_feat.shape[1]))
                            mask_feat[:, i, :] = rand_feat[:, rand_idx, :]

                        audio_mask[i] = 1.

                datum['a_feat'] = mask_feat.squeeze()
                datum['a_coord'] = audio_coord
                datum['a_mask'] = audio_mask

                label['audio_feat'] = audio_feat
                label['audio_label'] = audio_class.astype(np.long)
                label['audio_score'] = audio_score
                label['audio_coord'] = audio_coord
                label['audio_harmonics'] = audio_harmo
            else:
                datum['a_feat'] = audio_feat.squeeze()
                datum['a_coord'] = audio_coord


        if 'mask_lm' in self.pretrain_types:
            question = datum['question']
            mask_feat = question.copy()
            ques_mask = [0. for _ in range(len(mask_feat))]

            for i, token in enumerate(question):
                prob = random.random()

                if token in (self.tokenizer.cls_token_id, self.tokenizer.sep_token_id):
                    continue

                if prob < self.feature_mask_rate:
                    prob /= self.feature_mask_rate

                    if prob < 0.8:
                        mask_feat[i] = self.tokenizer.mask_token_id
                    elif prob < 0.9:
                        random_token = random.choice(batch_question_tokens)

                        # random token should not be CLS, SEP or MASK
                        while random_token in (self.tokenizer.cls_token_id,
                                               self.tokenizer.sep_token_id,
                                               self.tokenizer.mask_token_id,
                                               question[i]):
                            random_token = random.choice(batch_question_tokens)
                        mask_feat[i] = random_token

                    ques_mask[i] = 1.

            datum['l_feat'] = mask_feat
            datum['l_mask'] = ques_mask

            label['mask_lm'] = question
        else:
            datum['l_feat'] = datum['question']

        if 'qa' in self.pretrain_types:
            metadata['answer_str'] = datum['answer']
            # label['qa'] = one_hot_vectorize(datum['answer'], self.answer_label)
            if datum['answer'] in self.answer_label:
                label['qa'] = self.answer_label.index(datum['answer'])
                label['qa_valid'] = 1.
            else:
                label['qa'] = 0
                label['qa_valid'] = 0.

        datum.pop('answer')

        # Move metadata to label dict
        metadata['question'] = ' '.join(self.tokenizer.convert_ids_to_tokens(datum.pop('question')))
        metadata['question_id'] = datum.pop('question_id')
        metadata['question_type'] = datum.pop('question_type')
        metadata['video_id'] = datum.pop('video_id')
        for geo in ['cartesian', 'angular', 'spherical', 'quaternion']:
            if geo == self.geometry:
                label['ground'] = datum.pop(geo)[1:]
            else:
                datum.pop(geo)

        #print([f"{x:.3f}" for x in label['ground']], np.array(label['ground']).shape, np.array(datum['v_coord']).shape)

        return datum, label, metadata

    def collate_fn(self, batch):
        labels = []
        meta = []

        max_question_length = 0

        # follow the token distribution of curent batch
        batch_question_tokens = list(chain(*[datum['question'] for datum in batch])) 

        for i, datum in enumerate(batch):
            max_question_length = max(max_question_length, len(datum['question']))
            batch[i], label, metadata = self.prepare_pretrain(datum, batch_question_tokens)
            labels.append(label)
            meta.append(metadata)

        _batch = merge_dict(batch)
        _labels = merge_dict(labels)
        _meta = merge_dict(meta)
        
        # Debugging tokens
        # print('\n'.join(str(self.tokenizer.convert_ids_to_tokens(x)) for x in _batch['question']))
        for i, question in enumerate(_batch['l_feat']):
            surplus = max_question_length - len(question)

            _batch['l_feat'][i] += [self.tokenizer.pad_token_id for _ in range(surplus)]
            if "mask_lm" in self.pretrain_types:
                _batch['l_mask'][i] += [self.tokenizer.pad_token_id for _ in range(surplus)]

                _labels['mask_lm'][i] += [self.tokenizer.pad_token_id for _ in range(surplus)]
                # mlm = _labels['mask_lm'][i] + [self.tokenizer.pad_token_id for _ in range(surplus)]
                # _labels['mask_lm'][i] = np.zeros((len(mlm), self.model_config.vocab_size))
                # _labels['mask_lm'][i][np.arange(len(mlm)), mlm] = 1

        for k in _batch.keys():
            _batch[k] = np.array(_batch[k])
            if _batch[k].dtype == np.float64:
                _batch[k] = np.array(_batch[k], dtype=np.float32)

        for k in _labels.keys():
            _labels[k] = np.array(_labels[k])
            if _labels[k].dtype == np.float64:
                _labels[k] = np.array(_labels[k], dtype=np.float32)

        return _batch, _labels, _meta
