import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.ticker as ticker
import torch
from transformers import BertModel, BertTokenizer
from model.attention import attention

plt.rcParams['font.sans-serif'] = ['SimHei']




# 如果是两句话的输入[SEP]拼起来的，第二句话传入text_pair
def selfattn_visual(p_attn, text, text_pair=None):

    p_attn = p_attn.cpu().detach().numpy().squeeze(0)
    # print(p_attn.shape)


    if text_pair is not None:
        variables = ['cls'] + [str for str in text] + ['sep'] + [str for str in text_pair] + ['sep']
    else:
        variables = ['cls'] + [str for str in text] + ['sep']
    assert len(variables) == p_attn.shape[0]
    labels = variables

    df = pd.DataFrame(p_attn, columns=variables, index=labels)

    fig = plt.figure()

    ax = fig.add_subplot(111)

    cax = ax.matshow(df, interpolation='nearest', cmap='hot_r')
    fig.colorbar(cax)

    tick_spacing = 1
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))

    ax.set_xticklabels([''] + list(df.columns))
    ax.set_yticklabels([''] + list(df.index))

    plt.show()


if __name__ == '__main__':
    text1 = '今天天气怎么样？'
    text2 = '天气不太好，是雨天'

    model = BertModel.from_pretrained('bert-base-chinese')
    tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

    inputs = tokenizer(text=text1, text_pair=text2, return_tensors='pt')
    outputs = model(**inputs)
    sequence_outputs, cls = outputs[:2]
    outputs, p_attn = attention(query=sequence_outputs, key=sequence_outputs, value=sequence_outputs)

    selfattn_visual(p_attn, text=text1, text_pair=text2)