from torch.utils.data import Dataset, DataLoader
import numpy as np


def readDataFromFile(data_file):
    datas = []
    with open(data_file, 'r', encoding='utf-8') as reader:
        lines = reader.readlines()
        for line in lines:
            pair = line.strip().split('\t')
            datas.append(pair)
    return datas


class TrainData(Dataset):
    def __init__(self, data_file, max_length, tokenizer, model_type):
        self.datas = readDataFromFile(data_file)
        self.max_length = max_length
        self.tokenizer = tokenizer
        self.model_type = model_type

    def __getitem__(self, item):
        data = self.datas[item]
        query1, query2, label = data[0], data[1], int(data[2])
        tokenzied_dict = self.tokenizer.encode_plus(text=query1,
                                                    text_pair=query2,
                                                    max_length=self.max_length,
                                                    truncation=True,
                                                    padding='max_length')
        input_ids  = np.array(tokenzied_dict['input_ids'])
        attention_mask = np.array(tokenzied_dict['attention_mask'])
        if 'roberta' in self.model_type:
            return input_ids, attention_mask, np.array([label])
        token_type_ids = np.array(tokenzied_dict['token_type_ids'])
        return input_ids, token_type_ids, attention_mask, np.array([label]), query1, query2

    def __len__(self):
        return len(self.datas)




# if __name__ == '__main__':
#     from transformers import BertTokenizer, BertModel
#     tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#     model = BertModel.from_pretrained('bert-base-uncased')
#     train_dataset = TrainData(data_file='../data/MRPC/clean/train_clean.txt', max_length=100, tokenizer=tokenizer)
#     TrainDataLoader = DataLoader(dataset=train_dataset, batch_size=2, shuffle=True)
#     for batch in TrainDataLoader:
#         input_ids, token_type_ids, attention_mask, label = batch
#         outputs = model(input_ids=input_ids.long(), token_type_ids=token_type_ids.long(), attention_mask=attention_mask.long(), return_dict=True)
#         print(type(outputs))
#         res = outputs.pooler_output
#         res2 = outputs[1]
#         print(res.shape)
#         print(res == res2)
#         break

