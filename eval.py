import warnings
import hydra
import pytorch_lightning as pl
from transformers import BertTokenizer, logging
from data_module import ASTEDataModule
from module import ASTEModule
import os
import json

warnings.filterwarnings("ignore")
logging.set_verbosity_error()


def eval(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_tokens(["n't"])


    test_result = []

    trainer = pl.Trainer(gpus=args.gpus)
    load_file_path = 'model_ckpt'
    file_list = [f for f in os.listdir(load_file_path) if
                 os.path.isfile(os.path.join(load_file_path, f)) ]
    file_list = sorted(file_list)
    for file_name in file_list:
        args.dataset = file_name[:5]
        pl_data_module = ASTEDataModule(args, tokenizer)
        print("\n\n\n"+" Reloading model ".center(40, '='))
        print("File name:"+file_name)

        pl_module = ASTEModule.load_from_checkpoint(load_file_path+'\\'+file_name)
        test_result_temp = trainer.test(pl_module, datamodule=pl_data_module)
        test_result_temp[0].update({'run': file_name})
        test_result += test_result_temp
        print()
    #
    # json_str = json.dumps(test_result, indent=2)
    # with open('Jan23_500_all.json', 'w') as json_file:
    #     json_file.write(json_str)

@hydra.main(config_path='./config', config_name='main')
def main(conf):
    eval(conf)


if __name__ == '__main__':
    main()
