#ruled-based
when = ['什么时候', '何时']
how = ['如何', '怎么']

data_file = '../data/LCQMC/translation/en_split/en_split3.txt'

with open(data_file, 'r', encoding='utf-8') as reader:
    lines = reader.readlines()
    for line in lines:
        print(line.split('\t'))
        break