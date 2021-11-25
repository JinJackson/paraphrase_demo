from transformers import BertModel, AlbertModel, RobertaModel, BertPreTrainedModel, AlbertPreTrainedModel
from transformers import AutoModel, PreTrainedModel, AutoModelForPreTraining, AutoModelForQuestionAnswering
import torch.nn as nn

# BERT
class BertMatchModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        # 得到BERT输出的结果 ==> [seq_len, 768]
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        # 取出[CLS]表示, ==> [1, 768]
        pooler_output = outputs[1]
        # 把[CLS]表示送入linear分类得到打分 ==> [1, 1]
        logits = self.linear(self.dropout(pooler_output))
        # logits = self.linear(pooler_output)
        if labels is None:
            return logits
        # 交叉熵损失函数
        loss = self.loss_func(logits, labels.float())
        return loss, logits





# ERNIE
class ErnieMatchModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooler_output = outputs[1]
        logits = self.linear(self.dropout(pooler_output))
        # logits = self.linear(pooler_output)
        if labels is None:
            return logits
        loss = self.loss_func(logits, labels.float())
        return loss, logits

# Roberta
class RobertaMatchModel(BertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.roberta = RobertaModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, attention_mask, labels=None):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        pooler_output = outputs[1]
        logits = self.linear(self.dropout(pooler_output))
        if labels is None:
            return logits
        loss = self.loss_func(logits, labels.float())
        return loss, logits



class AlbertMatchModel(AlbertPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.albert = AlbertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.linear = nn.Linear(config.hidden_size, 1)
        self.loss_func = nn.BCEWithLogitsLoss()

    def forward(self, input_ids, token_type_ids, attention_mask, labels=None):
        outputs = self.albert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooler_output = outputs[1]
        logits = self.linear(self.dropout(pooler_output))
        if labels is None:
            return logits
        loss = self.loss_func(logits, labels.float())
        return loss, logits
