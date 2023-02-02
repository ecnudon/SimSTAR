import warnings
import hydra
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader
from transformers import BertTokenizer, logging
from module import ASTEModule
import json

warnings.filterwarnings("ignore")
logging.set_verbosity_error()



def generate_mask(seq_len):
    batch_size = len(seq_len)
    mask = torch.LongTensor(max(seq_len), batch_size).fill_(0)
    for idx, length in enumerate(seq_len):
        mask[:length, idx] = 1
    return mask

def collate_fn(sentence_set, tokenizer):
    batch_text = [sentence.split(' ') for sentence in sentence_set]
    input_ids, token_type_ids, attention_mask = tokenizer(
        batch_text,
        max_length=100,
        padding='longest',
        return_tensors='pt',
        is_split_into_words=True,
        truncation=True,
    ).values()
    tokenized_text = [tokenizer.tokenize(sen) for sen in sentence_set]
    batch_seq_len = [len(sen)+2 for sen in tokenized_text]
    seq_len = max(batch_seq_len)
    mask = generate_mask(batch_seq_len)

    return {
        "data":{
            'input_ids': input_ids,  # (batch_size, max_bert_len)
            'token_type_ids': token_type_ids,  # (batch_size, max_bert_len)
            'attention_mask': attention_mask,  # (batch_size, max_bert_len)
            'masks': mask,  # (max_bert_len, batch_size)
            'original_text': tokenized_text  # (batch_size, list)
        },
        "label":{
            "_":  torch.zeros(seq_len, seq_len, 1, 2),
            "__": torch.zeros(seq_len, seq_len, 1, 3)
        }
    }

def eval(args):
    trainer = pl.Trainer(gpus=args.gpus)
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(["n't"])
    load_file_path = 'model_ckpt/14res_Jan28_22.55.20_val_tri_f1=0.67159_epoch=99.ckpt'
    print(" Reloading model ".center(40, '='))

    pl_module = ASTEModule.load_from_checkpoint(load_file_path)

    while True:
        input_sentence = input('Input a review sentence(type "EXIT" for end):\n')
        # input_sentence = 'Nice but expensive Thai food, and seats are narrow.'
        if input_sentence == "EXIT":
            break
        single_set = [input_sentence]
        dataloader = DataLoader(
            dataset=single_set,
            batch_size=1,
            collate_fn=lambda x: collate_fn(x,tokenizer)
        )

        trainer.test(pl_module, dataloaders=dataloader, verbose=False)
        print("Triplets:")

        with open('_test_output_temp.json', 'r') as result:
            single_dict = json.load(result)[0]
            for tri in single_dict['triplets']:
                a,o,s = tri
                print(f"({a}, {o}, {s})", end=' ')
        print('\n')
@hydra.main(config_path='./config', config_name='main')
def main(conf):
    eval(conf)


if __name__ == '__main__':
    main()
