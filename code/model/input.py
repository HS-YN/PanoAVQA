import torch
from torch import nn


class BertEmbeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, model_config):
        super(BertEmbeddings, self).__init__()
        self.word_embeddings = nn.Embedding(model_config.vocab_size, model_config.hidden_size, padding_idx=0)
        self.position_embeddings = nn.Embedding(model_config.max_position_embeddings, model_config.hidden_size, padding_idx=0)
        self.token_type_embeddings = nn.Embedding(model_config.type_vocab_size, model_config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)
        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)

    def forward(self, input_ids, token_type_ids=None):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        if token_type_ids is None:
            token_type_ids = torch.zeros_like(input_ids)

        words_embeddings = self.word_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)

        embeddings = words_embeddings + position_embeddings + token_type_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class VisualFeatEncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.feature_fc = nn.Linear(model_config.visual_feat_dim, model_config.hidden_size)
        self.feature_ln = nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)

        self.coord_fc = nn.Linear(model_config.visual_coord_dim, model_config.hidden_size)
        self.coord_in = nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)

        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)

    def forward(self, v_input):
        feats, boxes = v_input

        feature_out = self.feature_ln(self.feature_fc(feats))
        coord_out = self.coord_in(self.coord_fc(boxes))
        output = self.dropout((feature_out + coord_out) / 2)

        return output


class VisualFeatNoCoordEncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()

        self.feat_fc = nn.Linear(model_config.visual_feat_dim, model_config.hidden_size)
        self.feat_ln = nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)
        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)

    def forward(self, v_input):
        feats, _ = v_input

        return self.dropout(self.feat_ln(self.feat_fc(feats)))


class VisualFeatConcatEncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        # AFAIK it is identical to VisualFeatEncoder, since it is mere decoupling of two fc
        # (Wx+a) + (Vy+b) = [W:V][x:y] + (a+b)
        # -> In fact, they are slightly different from VisualFeatEncoder due to nonlinearities
        self.feature = nn.Linear(model_config.visual_feat_dim + model_config.visual_coord_dim, model_config.hidden_size)
        self.layernorm = nn.LayerNorm(model_config.hidden_size, eps=model_config.layer_norm_eps)
        self.dropout = nn.Dropout(model_config.hidden_dropout_prob)

    def forward(self, v_input):
        v_input = torch.cat(v_input, -1)

        output = self.dropout(self.layernorm(self.feature(v_input)))
        return output


class AudioMonoEncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        f_dim = model_config.audio_feat_dim
        h_dim = model_config.hidden_size
        dropout_rate = model_config.hidden_dropout_prob
        eps = model_config.layer_norm_eps

        self.feat_fc = nn.Linear(f_dim, h_dim)
        self.feat_ln = nn.LayerNorm(h_dim, eps=eps)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, a_input):
        a_feat, _ = a_input
        output = self.dropout(self.feat_ln(self.feat_fc(a_feat)))
        return output


class AudioMonoTEncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        f_dim = model_config.audio_feat_dim
        h_dim = model_config.hidden_size
        dropout_rate = model_config.hidden_dropout_prob
        eps = model_config.layer_norm_eps

        self.feat_fc = nn.Linear(f_dim, h_dim)
        self.cord_fc = nn.Linear(2, h_dim)
        self.feat_ln = nn.LayerNorm(h_dim, eps=eps)
        self.cord_ln = nn.LayerNorm(h_dim, eps=eps)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, a_input):
        a_feat, a_cord = a_input
        feat_out = self.feat_ln(self.feat_fc(a_feat))
        cord_out = self.cord_ln(self.cord_fc(a_cord[:,:2]))

        return self.dropout((feat_out + cord_out) / 2)


class AudioMonoSEncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        f_dim = model_config.audio_feat_dim
        h_dim = model_config.hidden_size
        dropout_rate = model_config.hidden_dropout_prob
        eps = model_config.layer_norm_eps

        self.feat_fc = nn.Linear(f_dim, h_dim)
        self.cord_fc = nn.Linear(1, h_dim)
        self.feat_ln = nn.LayerNorm(h_dim, eps=eps)
        self.cord_ln = nn.LayerNorm(h_dim, eps=eps)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, a_input):
        a_feat, a_cord = a_input
        feat_out = self.feat_ln(self.feat_fc(a_feat))
        cord_out = self.cord_ln(self.cord_fc(a_cord[:,2]))

        return self.dropout((feat_out + cord_out) / 2)


class AudioMonoSTEncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        f_dim = model_config.audio_feat_dim
        h_dim = model_config.hidden_size
        dropout_rate = model_config.hidden_dropout_prob
        eps = model_config.layer_norm_eps

        self.feat_fc = nn.Linear(f_dim, h_dim)
        self.cord_fc = nn.Linear(3, h_dim)
        self.feat_ln = nn.LayerNorm(h_dim, eps=eps)
        self.cord_ln = nn.LayerNorm(h_dim, eps=eps)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, a_input):
        a_feat, a_cord = a_input
        feat_out = self.feat_ln(self.feat_fc(a_feat))
        cord_out = self.cord_ln(self.cord_fc(a_cord))

        return self.dropout((feat_out + cord_out) / 2)


class AudioStereoEncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        f_dim = model_config.audio_feat_dim
        h_dim = model_config.hidden_size
        dropout_rate = model_config.hidden_dropout_prob
        eps = model_config.layer_norm_eps
        
        self.left_fc = nn.Linear(f_dim, h_dim)
        self.righ_fc = nn.Linear(f_dim, h_dim)
        self.left_ln = nn.LayerNorm(h_dim, eps=eps)
        self.righ_ln = nn.LayerNorm(h_dim, eps=eps)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, a_input):
        a_feat, _ = a_input
        a_left = a_feat[:,0,:,:]
        a_righ = a_feat[:,1,:,:]

        left_out = self.left_ln(self.left_fc(a_left))
        righ_out = self.righ_ln(self.righ_fc(a_righ))
        return self.dropout((left_out + righ_out) / 2)


class AudioStereoSEncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        f_dim = model_config.audio_feat_dim
        h_dim = model_config.hidden_size
        dropout_rate = model_config.hidden_dropout_prob
        eps = model_config.layer_norm_eps

        self.left_fc = nn.Linear(f_dim, h_dim)
        self.righ_fc = nn.Linear(f_dim, h_dim)
        self.cord_fc = nn.Linear(1, h_dim)
        self.left_ln = nn.LayerNorm(h_dim, eps=eps)
        self.righ_ln = nn.LayerNorm(h_dim, eps=eps)
        self.cord_ln = nn.LayerNorm(h_dim, eps=eps)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, a_input):
        a_feat, a_cord = a_input
        a_left = a_feat[:,0,:,:]
        a_righ = a_feat[:,1,:,:]

        left_out = self.left_ln(self.left_fc(a_left))
        righ_out = self.righ_ln(self.righ_fc(a_righ))
        cord_out = self.cord_ln(self.cord_fc(a_cord[:,-1]))
        return self.dropout((left_out + righ_out + cord_out) / 3)


class AudioStereoTEncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        f_dim = model_config.audio_feat_dim
        h_dim = model_config.hidden_size
        dropout_rate = model_config.hidden_dropout_prob
        eps = model_config.layer_norm_eps

        self.left_fc = nn.Linear(f_dim, h_dim)
        self.righ_fc = nn.Linear(f_dim, h_dim)
        self.cord_fc = nn.Linear(2, h_dim)
        self.left_ln = nn.LayerNorm(h_dim, eps=eps)
        self.righ_ln = nn.LayerNorm(h_dim, eps=eps)
        self.cord_ln = nn.LayerNorm(h_dim, eps=eps)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, a_input):
        a_feat, a_cord = a_input
        a_left = a_feat[:,0,:,:]
        a_righ = a_feat[:,1,:,:]

        left_out = self.left_ln(self.left_fc(a_left))
        righ_out = self.righ_ln(self.righ_fc(a_righ))
        cord_out = self.cord_ln(self.cord_fc(a_cord[:,:-1]))
        return self.dropout((left_out + righ_out + cord_out) / 3)


class AudioStereoSTEncoder(nn.Module):
    def __init__(self, model_config):
        super().__init__()
        f_dim = model_config.audio_feat_dim
        h_dim = model_config.hidden_size
        dropout_rate = model_config.hidden_dropout_prob
        eps = model_config.layer_norm_eps

        self.left_fc = nn.Linear(f_dim, h_dim)
        self.righ_fc = nn.Linear(f_dim, h_dim)
        self.cord_fc = nn.Linear(3, h_dim)
        self.left_ln = nn.LayerNorm(h_dim, eps=eps)
        self.righ_ln = nn.LayerNorm(h_dim, eps=eps)
        self.cord_ln = nn.LayerNorm(h_dim, eps=eps)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, a_input):
        a_feat, a_cord = a_input
        a_left = a_feat[:,0,:,:]
        a_righ = a_feat[:,1,:,:]

        left_out = self.left_ln(self.left_fc(a_left))
        righ_out = self.righ_ln(self.righ_fc(a_righ))
        cord_out = self.cord_ln(self.cord_fc(a_cord))
        return self.dropout((left_out + righ_out + cord_out) / 3)
