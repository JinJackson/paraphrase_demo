data_file = '../data/LCQMC/clean/train_clean.txt'
translate_file = '../data/LCQMC/translation/train_en.txt'

from tqdm import tqdm

# 读取原始数据
def readDataFromFile(data_file):
    datas = []
    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            pair = line.strip().split('\t')
            datas.append(pair)
    return datas


# 读取原始数据的翻译数据
def readTranslateData(data_file):
    datas = []
    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            a_data = line.strip().split('||')
            # split bugs
            if len(a_data) == 2:
                line = line.replace('||', '|')
                a_data = line.strip().split('|')
            # delete ' '
            if len(a_data) == 4:
                for i in range(len(a_data)):
                    if not a_data[i] or a_data[i] == ' ':
                        a_data.pop(i)
                        break
            if len(a_data) != 3:
                # datas.append(a_data)
                datas.append(['UNK', 'UNK', 'UNK', 'UNK'])
            else:
                datas.append(a_data)
    return datas


# 标记数据的Qtype
def tagging_data(origin_datas, trans_datas):
    assert len(origin_datas) == len(trans_datas)
    all_taggings = []
    for origin_data, trans_data in tqdm(zip(origin_datas, trans_datas), total=len(origin_datas)):
        # import pdb
        # pdb.set_trace()
        origin_query1, origin_query2, label = origin_data
        if len(trans_data) == 4:
            all_taggings.append([origin_query1, origin_query2, label, ['UNK'], ['UNK']])
            continue
        try:
            trans_query1, trans_query2, _ = trans_data
            qtype_words = ['what', 'who', 'how', 'which', 'when', 'where', 'why']
            q_type1 = []
            q_type2 = []
            trans_query1 = trans_query1.lower()
            trans_query2 = trans_query2.lower()
            for qtype_word in qtype_words:
                if qtype_word in trans_query1:
                    q_type1.append(qtype_word)
                if qtype_word in trans_query2:
                    q_type2.append(qtype_word)
            else:
                if not q_type1:
                    q_type1 = ['others']
                if not q_type2:
                    q_type2 = ['others']
            all_taggings.append([origin_query1, origin_query2, label, q_type1, q_type2])
        except:
            all_taggings.append([origin_query1, origin_query2, label, ['UNK'], ['UNK']])
    return all_taggings


# # 写入数据
def written_file(labeled_data, written_file):
    dict_q_words = {'what': '事物', 'who': '人物', 'how': '做法', 'which': '选择', 'when': '时间', 'where': '地点', 'why': '原因',
                    'others': '其他', 'UNK': '未知'}
    with open(written_file, 'w', encoding='utf-8') as writer:
        for a_data in labeled_data:
            query1, query2, label, qmark1, qmark2 = a_data
            str_qmark1 = ''
            for q_word in qmark1:
                str_qmark1 += (dict_q_words[q_word] + ',')
            str_qmark2 = ''
            for q_word in qmark2:
                str_qmark2 += (dict_q_words[q_word] + ',')
            line = query1 + '[SEP]' + str_qmark1.rstrip(',') + '\t' + query2 + '[SEP]' + str_qmark2.rstrip(',') + '\t' + label + '\n'
            writer.write(line)


if __name__ == '__main__':
    data_file = '../data/BQ/clean/test_clean.txt'
    translate_file = '../data/BQ/translation/test_en.txt'
    written_filename = '../data/BQ/tagging/test_tag.txt'
    origin_datas = readDataFromFile(data_file)
    trans_datas = readTranslateData(translate_file)
    print(len(origin_datas), len(trans_datas))
    all_tagging_datas = tagging_data(origin_datas, trans_datas)
    print(len(all_tagging_datas))
    written_file(all_tagging_datas, written_filename)

