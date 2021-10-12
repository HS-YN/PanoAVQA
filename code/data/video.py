import json
import pickle as pkl
from tqdm import tqdm

import torch
import numpy as np

from exp import ex


# List of available (implemented) features
geometry_list = [None, 'cartesian', 'angular', 'spherical', 'quaternion']
visual_list = [None, 'rcnn_all', 'rcnn_center', 'rcnn_cube', 'rcnn_er', 'rcnn_nfov', 'i3d_center', 'i3d_er']
audio_list = [None, 'orig', 'pool_2', 'pool_4', 'pool_all', 'top_3', 'top_5']
stereo_list = ['bin', 'reg', 'raw'] # binary, regression, raw

audio_unit = 0.32
multiplier = {
    "orig": 1,
    "pool_2": 2,
    "pool_4": 4
}

@ex.capture()
def get_video(cache_path, rebuild_cache, feat_path, mode='train'):
    cache_path.mkdir(parents=True, exist_ok=True)

    visual, coordinate = get_visual(mode=mode)
    audio = get_audio()
    video = {
        "visual": visual,
        "coordinate": coordinate,
        "audio": audio
    }

    return video


@ex.capture()
def get_visual(feat_path, cache_path, rebuild_cache, data_path, model_config, mode='train'):
    visual_feature = model_config['visual_feature']
    geometry = model_config['geometry']

    visual_cache_file = f"{visual_feature}_{geometry}.pkl"
    #coordinate_cache_file = f"{geometry}.pkl"
    visual_cache_path = cache_path / visual_cache_file
    #coordinate_cache_path = cache_path / coordinate_cache_file

    assert visual_feature is None or (feat_path / 'visual' / visual_feature) in (feat_path / 'visual').glob('*'), \
        "[ERROR] video feature {} does not exist.".format(visual_feature)
    assert geometry in geometry_list, \
        "[ERROR] Geometry {} does not exist.".format(geometry)

    if rebuild_cache:
        if visual_cache_path.is_file():
            visual_cache_path.unlink()
        #if coordinate_cache_path.is_file():
        #    coordinate_cache_path.unlink()
    if visual_cache_path.is_file(): #and coordinate_cache_path.is_file():
        print(f'[LOG] Loading cached visual feautre from {visual_cache_path}...', end='', flush=True)
        visual, coordinate = torch.load(visual_cache_path)
        print('Complete!')
        #coordinate = torch.load(coordinate_cache_path)
    else:
        '''Reloading is inevitable unless both features exist'''
        visual, coordinate = _get_visual(visual_feature, geometry, feat_path, data_path, mode)
        torch.save((visual, coordinate), visual_cache_path)
        #torch.save(coordinate, coordinate_cache_path)

    return visual, coordinate


@ex.capture()
def get_audio(feat_path, cache_path, rebuild_cache, model_config):
    audio_feature = model_config['audio_feature']

    cache_file = f"{audio_feature}.pkl"
    path = cache_path / cache_file

    assert audio_feature is None or (feat_path / 'audio' / f'{audio_feature}_feat') in (feat_path / 'audio').glob('*'), \
        "[ERROR] audio feature {} does not exist.".format(audio_feature)

    if rebuild_cache and path.is_file():
        path.unlink()
    if path.is_file():
        print(f'[LOG] Loading cached audio feautre from {path}...', end='', flush=True)
        audio = torch.load(path)        
        print('Complete!')
    else:
        audio = _get_audio(audio_feature, feat_path)
        torch.save(audio, path)

    return audio


def _get_visual(visual_feature, geometry, feat_path, data_path, mode):
    '''Return requested visual feature'''
    visual_path = feat_path / 'visual'

    visual = {}
    coordinate = {} if geometry is not None else None

    if visual_feature is None:
        '''No visual feature'''
        return None, None

    metadata = json.load(open(data_path[mode], 'r'))['videos']

    if 'rcnn' in visual_feature:
        '''Visual features from RCNN family'''
        visual_path = visual_path / visual_feature

        for vid in tqdm(visual_path.glob('*'),
                        desc="Loading visual feature",
                        total=len(list(visual_path.glob('*')))):
            feats = pkl.load(open(vid, 'rb'))
            video_id = vid.stem.split('.')[0]

            embedding = feats['feat']
            score = feats['score']
            classes = feats['cls']
            if geometry == "cartesian":
                # Scale down cartesian coordinate w.r.t. width and height
                # RCNN utilized fixed width and height, thus we use
                w = 480. if visual_feature == "rcnn_center" else 1920.# float(metadata[video_id]['width'])
                h = 320. if visual_feature == "rcnn_center" else 1080.# float(metadata[video_id]['height'])
                feats[geometry] = [[g[0], g[1]/w, g[2]/h, g[3]/w, g[4]/h] for g in feats[geometry]]
                if len(feats[geometry]) == 0:
                    feats[geometry] = np.zeros((0, 5))
            elif geometry == 'spherical':
                feats[geometry][:,-2:] = feats['angular'][:,-2:]

            visual[video_id] = {
                "embedding": embedding,
                "score": np.array(score),
                "classes": np.array(classes),
                "coordinate": np.array(feats[geometry])
            }

            if geometry is not None:
                coordinate[video_id] = np.array(feats[geometry])
    elif 'i3d' in visual_feature:
        '''Visual features from I3D family'''
        visual_path = visual_path / visual_feature
        coordinate = None

        for vid in tqdm(visual_path.glob('*')):
            feats = pkl.load(open(vid, 'rb'))
            visual[video_id] ={
                "embedding": feats
            }
    else:
        assert False, "[ERROR] Unimplemented feature request in visual modality."

    return visual, coordinate


def _get_audio(audio_feature, feat_path):
    audio_path = feat_path / 'audio'

    audio = {}

    if audio_feature is None:
        return None

    stereo_path = audio_path / 'harmonics' / f'{audio_feature}_reg.json'
    stereo_feat = json.load(open(stereo_path, 'r'))

    if 'top' in audio_feature:
        time_feat = json.load(open(audio_path / f'{audio_feature}_time.json', 'r'))
    else:
        time_feat = {}

    for aud in tqdm((audio_path / f'{audio_feature}_feat').glob('*'),
                    desc="Loading audio feature",
                    total=len(list((audio_path / f'{audio_feature}_feat').glob('*')))):
        feats = pkl.load(open(aud, 'rb'))
        video_id = aud.stem.split('.')[0]
        label = pkl.load(open(audio_path / f'{audio_feature}_label' / f'{video_id}.pkl', 'rb'))
        harmonics = stereo_feat[video_id][:len(label)]
        audio_len = pkl.load(open(audio_path / f'orig_label' / f'{video_id}.pkl', 'rb')).shape[0] * audio_unit

        score = torch.sigmoid(torch.from_numpy(label)).numpy()
        label = np.argmax(score, axis=1)
        score = np.max(score, axis=1)

        if 'top' not in audio_feature:
            a_start = np.array([audio_unit * multiplier[audio_feature] for _ in range(label.shape[0]+1)])
        else:
            a_start = np.array([0.] + time_feat[video_id])
        a_duration = a_start[1:] - a_start[:-1]
        a_start = a_start[:-1]
        a_coord = np.array([[a_start[i], a_duration[i], harmonics[i]] for i in range(a_start.shape[0])])

        audio[video_id] = {
            "embedding": feats,
            "score": score,
            "classes": label,
            "harmonics": np.array(harmonics),
            "coordinate": a_coord
        }

    return audio
