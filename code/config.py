from copy import deepcopy

default_args = {
    # Logging and general configuration
    'debug': True,
    'num_workers': 40,
    'random_seed': 1234,
    'log_path': 'data/log',
    'ckpt_path': 'data/ckpt',
    'ckpt_name': None,
    'log_tag': '',                    # brief tagging for readibility
    'log_keys': [
        'log_tag',
        'model_name'
    ],
    'feat_path': 'data/feat',
    'pretrain_path': 'data/feat/label/trainval.json',
    'train_path': 'data/feat/label/trainval.json',
    'preval_path': 'data/feat/label/test.json',
    'val_path': 'data/feat/label/test.json',
    'test_path': 'data/feat/label/test.json',
    'answer_path': 'data/feat/label/answer_2020.json',
    'cache_path': 'data/cache',
    'output_path': 'data/output',
    'rebuild_cache': False,

    'model_name': 'lavit',
    'transformer_name': 'bert',
    'num_objects': 36,

    # Learning configuration
    'pretrain_epochs': 3,
    'max_epochs': 10,

    'batch_size': 32,
    'grad_acc_steps': 4,

    'split_train': False,
    'optimizer_name': 'bert_adam',
    'scheduler_name': 'linear', 
    'weight_decay': 1e-2,
    'learning_rate': 5e-5,
    'pretrain_learning_rate': 1e-4,
    'transformer_learning_rate': 1e-5,
    'warmup': 0.1,
    'feature_mask_rate': 0.15,
    'device': 'cuda',

    'max_length_qa': 60,

    'model_config': {
        # Input/output features
        'audio_feature': 'top_3',        # None,orig,pool_{2,4,all},top_{3,5}
        'visual_feature': 'rcnn_all',    # None,rcnn_all,rcnn_center,rcnn_cube,rcnn_er,rcnn_nfov,i3d_center,i3d_er
        'geometry': 'quaternion',        # cartesian,angular,spherical,quaternion

        # Input/output dimension
        'vocab_size': 30522,
        'num_answers': 2020,
        'visual_feat_dim': 2048,
        'visual_label_dim': 200,
        'visual_coord_dim': 6,          
        'audio_feat_dim': 2048,
        'audio_label_dim': 527,
        'audio_coord_dim': 3,

        # Model embedding dimension
        'hidden_size': 768,
        'num_attention_heads': 12,
        'intermediate_size': 3072,
        'max_position_embeddings': 512,
        'type_vocab_size': 2,
        'pad_token_id': 0,

        # Probability / activation
        'hidden_act': 'gelu',
        'hidden_dropout_prob': 0.1,
        'attention_probs_dropout_prob': 0.1,
        'initializer_range': 0.02,
        'layer_norm_eps': 1e-12,
        'loss_normalizer': 6.67,        # 1 / 0.15

        # Model structure 
        'use_concat_encoder': False,
        'use_concat_decoder': True,
        'use_stereo_audio': True,
        'use_cls_token': True,
        'use_grounding': True,
        'l_layers': 9,
        'v_layers': 5,
        'a_layers': 5,
        'x_layers': 5,

        # Pretrain task toggle
        'mask_lm': True,
        'visual_feat': True,
        'visual_coord': True,
        'visual_label': True,
        'audio_feat': True,
        'audio_coord': True,
        'audio_label': True,
        'vl_match': True,
        'al_match': True,
        'qa': True,
        'ground': True,

        # Hyperparameter
        'lambda_ground': 0.2
    },
}

lxmert_args = deepcopy(default_args)
lxmert_args['model_config'].update({
        # Pretrain task toggle
        'mask_lm': True,
        'visual_feat': True,
        'visual_coord': False,
        'visual_label': True,
        'audio_feat': False,
        'audio_coord': False,
        'audio_label': False,
        'vl_match': True,
        'al_match': False,
        'qa': True,
        'ground': False,
    })
lxmert_args.update({
        'use_cls_token': False,
        'use_concat_decoder': False,
        'use_grounding': False,
    })


args_dict = {
    'lxmert': lxmert_args,
    'lavit': default_args,
    'bert': default_args
}