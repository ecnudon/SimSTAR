import torch
import torch.nn as nn
from transformers import BertModel
from transformers.modeling_outputs import ModelOutput

class SpanRepresentation(nn.Module):
    """
    test.
    """
    def __init__(self, config):
        super(SpanRepresentation, self).__init__()

        self.config = config
        self.input_dim = config.hidden_size  * 2 + config.span_len_embed_dim
        self.target_dim = config.span_feature_dim
        self.feature2hidden = nn.Linear(self.input_dim, self.target_dim)
        self.span_len_embed = nn.Embedding(2 * config.max_length, config.span_len_embed_dim)

    def forward(
            self,
            feature_states,
    ):
        '''
        This function used to get the
        :param feature_states: (batch_size, seq_len, hidden_size)
        :return:
        '''

        features = feature_states.permute(1, 0, 2) #(seq_len, batch_size, hidden_size)
        seq_len, batch_size, _ = features.size()

        start = features.unsqueeze(1).repeat(1, seq_len, 1, 1)  #(seq_len, repeated_seq_len, batch_size, hidden_size)
        end = features.unsqueeze(0).repeat(seq_len, 1, 1, 1)    #(repeated_seq_len, seq_len, batch_size, hidden_size)
        # span_matrix = torch.cat((start,end), dim=-1)

        span_index = torch.arange(seq_len).repeat(seq_len, 1).cuda()
        idx_start = span_index.unsqueeze(0).repeat(batch_size, 1, 1)
        idx_end = span_index.T.unsqueeze(0).repeat(batch_size, 1, 1)
        idx_distance = (idx_start - idx_end).permute(1, 2, 0)
        idx_distance = idx_distance + torch.IntTensor(idx_distance.size()).fill_(self.config.max_length).cuda()
        span_len_embedding = self.span_len_embed(idx_distance)
        span_matrix = torch.cat((start,end, span_len_embedding), dim=-1)
        # span_matrix:(seq_len, seq_len, batch_size, 1 or 2 * hidden_size+ )
        span_matrix = self.feature2hidden(span_matrix)

        return span_matrix #(seq_len, seq_len, batch_size, span_feature_dim)

class TermClassifier(nn.Module):
    def __init__(self, config):
        super(TermClassifier, self).__init__()

        self.term2idx = config.term2idx
        self.feature_dim = config.span_feature_dim
        self.term_types = len(self.term2idx)
        self.transform = nn.Linear(self.feature_dim, self.term_types)
        self.layer_norm = nn.LayerNorm(self.feature_dim)
        self.dropout = nn.Dropout(p=config.dropout)
        self.elu = nn.ELU()

    def forward(
            self,
            span_matrix,
            masks
    ):
        '''
        From the ``span_matrix`` of (seq_len, seq_len, batch_size, span_feature_dim) to soft_labels
        :param input_matrix:
        :return:
        '''

        # span_matrix: (seq_len, seq_len, batch_size, span_feature_dim)
        seq_len, _, batch_size, __ = span_matrix.size()
        spans = self.layer_norm(span_matrix)
        spans = self.elu(self.dropout(spans))
        soft_label = torch.sigmoid(self.transform(spans))

        # NER has to keep the order of (start, end), so the matrix has to be upper triangular
        diagonal_mask = torch.triu(torch.ones(batch_size, seq_len, seq_len)).cuda()
        diagonal_mask = diagonal_mask.permute(1, 2, 0)

        #masks: (seq_len, batch_size)
        mask_s = masks.unsqueeze(1).repeat(1, seq_len, 1)
        mask_e = masks.unsqueeze(0).repeat(seq_len, 1, 1)

        mask_ner = mask_s * mask_e
        masks = diagonal_mask * mask_ner
        masks = masks.unsqueeze(-1).repeat(1, 1, 1, self.term_types)

        soft_label = soft_label * masks

        return soft_label

class SentiClassifier(nn.Module):
    def __init__(self, config):
        super(SentiClassifier, self).__init__()

        self.senti2idx = config.senti2idx
        self.feature_dim = config.span_feature_dim
        self.senti_types = len(self.senti2idx)
        self.transform = nn.Linear(self.feature_dim, self.senti_types)
        self.layer_norm = nn.LayerNorm(self.feature_dim)
        self.dropout = nn.Dropout(p=config.dropout)
        self.elu = nn.ELU()

    def forward(
            self,
            span_matrix,
            masks
    ):
        '''
        From the ``span_matrix`` of (seq_len, seq_len, batch_size, span_feature_dim) to soft_labels
        :param input_matrix:
        :return:
        '''

        # span_matrix: (seq_len, seq_len, batch_size, span_feature_dim)
        seq_len, _, batch_size, __ = span_matrix.size()
        spans = self.layer_norm(span_matrix)
        spans = self.elu(self.dropout(spans))
        soft_label = torch.sigmoid(self.transform(spans))

        #masks: (seq_len, batch_size)
        mask_e1 = masks.unsqueeze(1).repeat(1, seq_len, 1)
        mask_e2 = masks.unsqueeze(0).repeat(seq_len, 1, 1)

        masks = mask_e1 * mask_e2
        masks = masks.unsqueeze(-1).repeat(1, 1, 1, self.senti_types)

        soft_label = soft_label * masks

        return soft_label

class SimSTAR(nn.Module):
    def __init__(self, config):
        super(SimSTAR, self).__init__()

        self.bert = BertModel.from_pretrained(config.model_name_or_path)
        self.feature_extractor = SpanRepresentation(config)
        self.term_classifier = TermClassifier(config)
        self.senti_classifier = SentiClassifier(config)
        self.dropout = nn.Dropout(config.dropout)

    def forward(
            self,
            input_ids,
            token_type_ids,
            attention_mask,
            masks,
            original_text
    ):

        bert_output = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        features = self.feature_extractor(bert_output['last_hidden_state'])
        term_pred = self.term_classifier(features, masks)
        senti_pred = self.senti_classifier(features, masks)

        return term_pred, senti_pred

    def resize_token_embeddings(self, new_num_tokens):
        return self.bert.resize_token_embeddings(new_num_tokens)
