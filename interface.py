from model.MatchModel import BertMatchModel
from transformers import BertTokenizer
import math
import sys



def sigmoid(x):
    return 1 / (1 + pow(math.e, -x))


def load_model(model_path):
    model = BertMatchModel.from_pretrained(model_path)
    tokenizer = BertTokenizer.from_pretrained(model_path)
    return model, tokenizer


def convert_input(text1, text2, tokenizer):
    max_length = 256
    tokenized_dict = tokenizer.encode_plus(text=text1,
                                           text_pair=text2,
                                           truncation=True,
                                           padding='max_length',
                                           max_length=max_length,
                                           return_tensors='pt')
    input_ids = tokenized_dict['input_ids']
    token_type_ids = tokenized_dict['token_type_ids']
    attention_mask = tokenized_dict['attention_mask']
    return input_ids, token_type_ids, attention_mask


def predict(model_path, text1, text2):
    model, tokenizer = load_model(model_path)
    input_ids, token_type_ids, attention_mask = convert_input(text1=text1,
                                                              text2=text2,
                                                              tokenizer=tokenizer)
    logits = model(input_ids=input_ids,
                   token_type_ids=token_type_ids,
                   attention_mask=attention_mask).squeeze(0).detach().cpu().tolist()[0]
    if logits >= 0:
        return sigmoid(logits), True
    if logits < 0:
        return sigmoid(logits), False


if __name__ == '__main__':
    print('start')
    # 需要预测的两个文本
    text1 = sys.argv[1]
    text2 = sys.argv[2]

    # 加载的训练好的模型路径
    model_path = sys.argv[3]

    probability, result = predict(model_path=model_path,
                                  text1=text1,
                                  text2=text2)

    print("您输入的两个文本为复述的概率为%.2f, 本AI认为结果为%s" % (probability * 100, str(result)))
