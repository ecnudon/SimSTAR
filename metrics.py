import torch


class metrics:

    def __init__(self, args):
        self.term2idx = args.term2idx
        self.senti2idx = args.senti2idx
        self.aspect_threshold = 0.4
        self.opinion_threshold = 0.4
        self.senti_threshold = 0.45
        self.value_targets = ['_p', '_r', '_f1']

    def get_right_term_pair(self, term_pred, term_gold):
        # term_gold : seq_len * seq_len * batch_size * term_type

        right_terms = term_gold * term_pred
        right_terms = torch.sum(right_terms, dim=1)

        right_terms = torch.sum(right_terms, dim=-1)
        right_terms = self._binary_format(right_terms, threshold=0)  # (seq_len, batch_size)

        seq_len = term_gold.size(0)
        e1 = right_terms.unsqueeze(0).repeat(seq_len, 1, 1)
        e2 = right_terms.unsqueeze(1).repeat(1, seq_len, 1)
        right_terms = e1 * e2
        right_terms = right_terms.unsqueeze(-1).repeat(1, 1, 1, len(self.senti2idx))
        return right_terms

    def terms_pair_filter(self, term_pred, senti_pred):
        '''
        if the model predicts that there is a sentiment between position A and B,
        unless the model also *predicts* out spans which start at A and B respectively,
        this sentiment is irregular and should be filter out.
        '''
        # sum up the span_end dim. If there is any span START at i, the i th dim = 1
        term_mask = torch.sum(term_pred, dim=1)
        term_mask = torch.sum(term_mask, dim=-1)
        # aspects, opinions = torch.chunk(term_mask, 2, dim=-1)
        # add up may cause element > 1. Format it.
        term_mask = self._binary_format(term_mask, threshold=0)

        seq_len = term_mask.size(0)
        e1 = term_mask.unsqueeze(0).repeat(seq_len, 1, 1)
        e2 = term_mask.unsqueeze(1).repeat(1, seq_len, 1)
        term_mask = e1 * e2
        term_mask = term_mask.unsqueeze(-1).repeat(1, 1, 1, len(self.senti2idx))

        complete_senti_pred = senti_pred * term_mask
        return complete_senti_pred

    def count_tri_num(self, term_pred, term_gold, senti_pred, senti_gold):
        aspects, opinions = torch.chunk(term_pred, 2, dim=-1)
        aspects = self._binary_format(aspects, threshold=self.aspect_threshold)
        opinions = self._binary_format(opinions, threshold=self.opinion_threshold)
        term_pred = torch.cat((aspects, opinions), dim=-1)

        senti_pred = self._binary_format(senti_pred, threshold=self.senti_threshold)


        term_right_mask = self.get_right_term_pair(term_pred, term_gold)
        senti_pred = self.terms_pair_filter(term_pred, senti_pred)
        senti_pred_num = senti_pred.sum().item()

        senti_gold_num = senti_gold.sum().item()

        senti_right = senti_pred * senti_gold
        # right predicted senti should filtrate
        senti_right = senti_right * term_right_mask
        right_num = senti_right.sum().item()

        tri_count_list = [senti_pred_num, senti_gold_num, right_num]
        return tri_count_list


    def count_tri_num_and_display(self, term_pred, term_gold, senti_pred, senti_gold, original_text):
        aspects, opinions = torch.chunk(term_pred, 2, dim=-1)
        aspects = self._binary_format(aspects, threshold=self.aspect_threshold)
        opinions = self._binary_format(opinions, threshold=self.opinion_threshold)
        term_pred = torch.cat((aspects, opinions), dim=-1)
        batch_term = self.display_term(term_pred, original_text)

        senti_pred = self._binary_format(senti_pred, threshold=self.senti_threshold)

        term_right_mask = self.get_right_term_pair(term_pred, term_gold)
        senti_pred = self.terms_pair_filter(term_pred, senti_pred)
        batch_tri = self.display_tri(senti_pred, batch_term, original_text)

        senti_pred_num = senti_pred.sum().item()
        senti_gold_num = senti_gold.sum().item()

        senti_right = senti_pred * senti_gold
        # right predicted senti should filtrate
        senti_right = senti_right * term_right_mask
        right_num = senti_right.sum().item()

        tri_count_list = [senti_pred_num, senti_gold_num, right_num]

        return tri_count_list, batch_tri

    def count_term_num(self, term_pred, term_gold):

        aspects, opinions = torch.chunk(term_pred, 2, dim=-1)
        aspects = self._binary_format(aspects, threshold=self.aspect_threshold)
        opinions = self._binary_format(opinions, threshold=self.opinion_threshold)
        term_pred = torch.cat((aspects, opinions), dim=-1)

        # for term_type, idx in self.term2idx.items():
        term_pred_num = term_pred.sum().item()
        term_gold_num = term_gold.sum().item()
        term_right = term_pred * term_gold
        term_right_num = term_right.sum().item()
        term_count_list = [term_pred_num, term_gold_num, term_right_num]

        return term_count_list

    def _binary_format(self, tensor, threshold: float):
        return torch.where(tensor > threshold,
                           torch.ones_like(tensor),
                           torch.zeros_like(tensor))

    def calculate(self, pred_num, gold_num, right_num):
        precision = float(right_num) / pred_num if pred_num > 0 else 0
        recall = float(right_num) / gold_num if gold_num > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0

        return precision, recall, f1

    def display_term(self, term_pred, original_text):
        idx2term = { v:k for k,v in self.term2idx.items()}
        term_pred = term_pred.permute(2, 0, 1, 3)
        ones_select = torch.nonzero(term_pred).tolist()
        batch_term_result = [[] for _ in range(len(original_text))]
        for one in ones_select:
            sentence_idx, head, tail, term_idx = one
            # Due to the [CLS] in bert tokenization, we minus 1 on indices
            span_text = ' '.join(original_text[sentence_idx][head-1:tail]).replace(' ##', '')
            batch_term_result[sentence_idx].append(
                [idx2term[term_idx], head, tail, span_text]
            )
        return batch_term_result

    def display_tri(self, senti_pred, batch_term, original_text):
        idx2senti = { v:k for k,v in self.senti2idx.items()}
        senti_pred = senti_pred.permute(2, 0, 1, 3)
        ones_select = torch.nonzero(senti_pred).tolist()
        batch_tri_result = [[] for _ in range(len(batch_term))]
        for one in ones_select:
            sentence_idx, a_head, o_head, senti_idx = one
            asp_set, opi_set = [], []
            for term in batch_term[sentence_idx]:
                if term[0] == 'aspect' and term[1] == a_head:
                    asp_set.append(term)
                elif term[0] == 'opinion' and term[1] == o_head:
                    opi_set.append(term)

            if len(asp_set)>0 and len(opi_set)>0:
                if len(asp_set) > 1:
                    sorted(asp_set, key=lambda lst:lst[2])
                if len(opi_set) > 1:
                    sorted(opi_set, key=lambda lst:lst[2])
                asp, opi = asp_set[0], opi_set[0]
                batch_tri_result[sentence_idx].append(
                    [asp[3], opi[3], idx2senti[senti_idx]])

        return [{'sentence':' '.join(sent).replace(' ##',''), 'triplets':tri_list}
                for sent, tri_list in zip(original_text, batch_tri_result)]



