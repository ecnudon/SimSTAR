from torch.utils.data import Dataset
import ast
import os


class ASTEDataset(Dataset):
    def __init__(self, args, mode):
        super(ASTEDataset, self).__init__()
        self.data = []
        self.file_path = os.path.join(args.data_dir, args.dataset, args.data_file_format)

        if mode in ['train', 'dev', 'test']:
            self.file_path = self.file_path.replace('xxx', mode)
            self.prepare_data(self.file_path)


    def prepare_data(self, file):
        with open(file, 'r', encoding='utf8', errors='ignore') as f:
            for data in f.readlines():
                sentence, tri_list = data.rstrip('\n').split('####')
                tokens = sentence.split(' ')
                tri_list = ast.literal_eval(tri_list)
                single_sent_tri = [
                    {
                        'aspect': [asp[0], asp[-1]],
                        'opinion': [opi[0], opi[-1]],
                        'sentiment': senti
                    }
                    for asp, opi, senti in tri_list
                ]

                self.data.append({
                    'text_list': tokens,
                    'triple_list': single_sent_tri
                })


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
