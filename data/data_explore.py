import ast
import os

DATASETS = ['14lap', '14res', '15res', '16res']
FILES = ['train', 'dev', 'test']

print(os.getcwd())
maxlen = 0
for dataset in DATASETS:
    for file in FILES:
        sc,token_count, asp_count,opi_count, tri_count = 0,0,0,0,{"POS":0,"NEU":0,"NEG":0}
        ori_path = os.path.join('ASTE-Data-V2-EMNLP2020',dataset, file+'_triplets.txt' )
        with open(ori_path, 'r', encoding='utf8', errors='ignore') as f:
            lines = f.readlines()
            for data in lines:
                asp_set, opi_set = [], []
                sentence, tri_list = data.rstrip('\n').split('####')
                tokens = sentence.split(' ')
                if len(tokens)==83:
                    print(data)
                token_count += len(tokens)**2
                maxlen = max(maxlen,len(tokens))
                sc += len(tokens)
                tri_list = ast.literal_eval(tri_list)
                for asp, opi, senti in tri_list:
                    asp_set.append(asp[0])
                    opi_set.append(opi[0])
                    tri_count[senti] = tri_count[senti] + 1
                asp_count += len(set(asp_set))
                opi_count += len(set(opi_set))
            # print(dataset, file)
            # print('total span:',token_count,'  tA:',asp_count,'  tO:',opi_count,'  tT:',tri_count)
            # print('the file has ', len(lines), 'sentences.')
            # print(f'avg: {(token_count/len(lines)):.1f} spans per sentence,\n'
            #       f' {(sc/len(lines)):.1f} tokens per sentence,\n\n')

            # print(f'& {file.capitalize()} & {len(lines)} & {(sc/len(lines)):.1f} & {(token_count/len(lines)):.1f} & '
            #       f'{asp_count} & {opi_count} & {tri_count["POS"]} & {tri_count["NEU"]} & {tri_count["NEG"]}\\\\')
print(maxlen)