from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from dataset import ASTEDataset
import torch
import copy

def generate_term_labels(max_len, tri_list, term2idx):
    '''
    Normal idea, judge entities with span representation,
    '''
    labels = torch.FloatTensor(max_len, max_len, len(term2idx)).fill_(0)
    for tri in tri_list:
        for term, idx in term2idx.items():
            head, tail = tri[term]
            labels[head][tail][idx] = 1

    return labels


def generate_senti_labels(max_len, tri_list, senti2idx):
    '''
    Normal idea, judge senti as relation with span representation, label is at the biggest span
    '''
    labels = torch.FloatTensor(max_len, max_len, len(senti2idx)).fill_(0)
    for tri in tri_list:
        a_head, _ = tri['aspect']
        o_head, _ = tri['opinion']
        labels[a_head][o_head][ senti2idx[tri['sentiment']] ] = 1

    return labels


def generate_mask(seq_len):
    batch_size = len(seq_len)
    mask = torch.LongTensor(max(seq_len), batch_size).fill_(0)
    for idx, length in enumerate(seq_len):
        mask[:length, idx] = 1
    return mask


def collate_fn(data, tokenizer, truncation_length, term2idx, senti2idx):
    '''
    **IMPORTANT!**
    We use the "longest" padding scheme, so first batchify then tokenize
    should be the correct order. Therefore, we tokenize the batch here in `collate_fn`.
    '''
    batch_text = [item['text_list'] for item in data]
    input_ids, token_type_ids, attention_mask = tokenizer(
        batch_text,
        max_length=truncation_length,
        padding='longest',
        return_tensors='pt',
        is_split_into_words=True,
        truncation=True,
    ).values()
    tokenized_text = [tokenizer.tokenize(' '.join(sen)) for sen in batch_text]

    batch_tri_list = [item['triple_list'] for item in data]
    batch_seq_len = [item['token_nums'] for item in data]
    max_bert_len = max(batch_seq_len)

    term_label = [generate_term_labels(max_bert_len, tri_list, term2idx) for tri_list in batch_tri_list]
    senti_label = [generate_senti_labels(max_bert_len, tri_list, senti2idx) for tri_list in batch_tri_list]
    mask = generate_mask(batch_seq_len)

    return {
        'data':{
            'input_ids': input_ids,  # (batch_size, max_bert_len)
            'token_type_ids': token_type_ids,  # (batch_size, max_bert_len)
            'attention_mask': attention_mask,  # (batch_size, max_bert_len)
            'masks': mask,  # (max_bert_len, batch_size)
            'original_text': tokenized_text # (batch_size, list)
        },
        'label':{
            'term_labels': torch.stack(term_label, dim=2), # (max_bert_len, max_bert_len, batch_size, len(term2idx) )
            'senti_labels': torch.stack(senti_label, dim=2), # (max_bert_len, max_bert_len, batch_size, len(senti2idx) )

        }
    }

class ProcessedDataset(Dataset):
    def __init__(self, dataset, process_fn):
        self.dataset = dataset
        self.process_fn = process_fn

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        return self.process_fn(self.dataset[item])


class ASTEDataModule(pl.LightningDataModule):
    def __init__(self, args, tokenizer):
        super(ASTEDataModule, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.data_process = self.bert_preprocess_fn # can easily change here
        self.test_dataset = ProcessedDataset(ASTEDataset(args, 'test'), self.data_process)
        self.train_dataset = ProcessedDataset(ASTEDataset(args, 'train'), self.data_process)
        self.dev_dataset = ProcessedDataset(ASTEDataset(args, 'dev'), self.data_process)

        self.term2idx = args.term2idx
        self.senti2idx = args.senti2idx

    def bert_preprocess_fn(self, item):
        text_list = item['text_list']
        triple_list = copy.deepcopy(item['triple_list'])
        token_nums = 0
        # New list for return

        # if no subwords in tokenization
        if len(text_list) == len(self.tokenizer.tokenize(' '.join(text_list))):
            # idx in triple_list only need to add 1 for [CLS] token
            self.all_position_add(triple_list, 1) # [CLS] token
            token_nums = len(text_list) + 2 # [CLS] [SEP]

        # else: existing subword-split, need to rebuild the index totally
        else:
            bert_input_seq = []
            bert_input_seq.append(self.tokenizer.cls_token)
            idx_mapping_for_start = {}
            idx_mapping_for_end = {}
            for idx, word in enumerate(text_list):
                idx_mapping_for_start[idx] = len(bert_input_seq)
                tokenized = self.tokenizer.tokenize(word)
                bert_input_seq += tokenized
                idx_mapping_for_end[idx] = len(bert_input_seq) - 1
            bert_input_seq.append(self.tokenizer.sep_token)

            # bert_input_seq is unused, just for checking
            token_nums = len(bert_input_seq)
            # you can use `assert` for checking.

            for tri in triple_list:
                a_head, a_tail = tri['aspect']
                o_head, o_tail = tri['opinion']
                tri['aspect'] = [idx_mapping_for_start.get(a_head), idx_mapping_for_end.get(a_tail)]
                tri['opinion'] = [idx_mapping_for_start.get(o_head), idx_mapping_for_end.get(o_tail)]

        return {
            'text_list': text_list,
            'triple_list': triple_list,
            'token_nums' : token_nums
        }


    def all_position_add(self, tri_list, to_add_num):
        for tri in tri_list:
            tri['aspect'] = [ idx+to_add_num for idx in tri['aspect'] ]
            tri['opinion'] = [idx + to_add_num for idx in tri['opinion'] ]
        return tri_list


    def train_dataloader(self):
        return DataLoader(
            dataset=self.train_dataset,
            batch_size=self.args.batch_size,
            shuffle=True,
            collate_fn=lambda x: collate_fn(x, self.tokenizer, self.args.max_length,  self.term2idx, self.senti2idx)
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.args.batch_size,
            collate_fn=lambda x: collate_fn(x, self.tokenizer, self.args.max_length,  self.term2idx, self.senti2idx)
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dev_dataset,
            batch_size=self.args.batch_size,
            collate_fn=lambda x: collate_fn(x, self.tokenizer, self.args.max_length,  self.term2idx, self.senti2idx)
        )
