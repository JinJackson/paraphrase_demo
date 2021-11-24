from all_dataset import readDataFromFile
data_file = '../data/BQ/clean/test_clean.txt'
write_file_direc = '../data/BQ/translation/origin_split/test/'

max_length = 250 #每个文件最长长度数量
all_datas = readDataFromFile(data_file)

if len(all_datas) % max_length == 0:
    nums = len(all_datas) // max_length
else:
    nums = (len(all_datas) // max_length) + 1

for i in range(0, nums):
    with open(write_file_direc+'test_split'+str(i+1)+'.txt', 'w', encoding='utf-8') as writer:
        start = i * max_length
        write_datas = all_datas[start:start+max_length]
        for a_data in write_datas:
            query1, query2, label = a_data
            writer.write(query1 + ' || ' + query2 + '||' + str(label) + '\n')