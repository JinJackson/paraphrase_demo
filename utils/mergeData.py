# 更改一下文件数量！！！！！！！！！！！！！！
nums = 400

data_file_path = '../data/BQ/translation/en_split/train/'
written_file = '../data/BQ/translation/train_en.txt'

all_datas = []
wrong_data = []
for i in range(0, nums):
    with open(data_file_path + 'train_split' + str(i + 1) + '.txt', 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for j in range(len(lines)):
            line = lines[j]
            all_datas.append(line)
#
with open(written_file, 'w', encoding='utf-8') as writer:
    for a_data in all_datas:
        writer.write(a_data.strip()+'\n')