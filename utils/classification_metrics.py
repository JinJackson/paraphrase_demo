# 输入是模型输出的预测分数
def accuracy(all_logits, all_labels):
    all_predict = (all_logits > 0) + 0
    results = (all_predict == all_labels)
    acc = results.sum() / len(all_predict)
    return acc


def precision(all_logits, all_labels):
    all_pred = (all_logits > 0) + 0
    TP = ((all_pred == 1) & (all_labels == 1)).sum()
    FP = ((all_pred == 1) & (all_labels == 0)).sum()
    precision = TP / (TP + FP)
    return precision


def recall(all_logits, all_labels):
    all_pred = (all_logits > 0) + 0
    TP = ((all_pred == 1) & (all_labels == 1)).sum()
    FN = ((all_pred == 0) & (all_labels == 1)).sum()
    recall = TP / (TP + FN)
    return recall


def f1_score(all_logits, all_labels):
    all_pred = (all_logits > 0) + 0
    TP = ((all_pred == 1) & (all_labels == 1)).sum()
    FP = ((all_pred == 1) & (all_labels == 0)).sum()
    FN = ((all_pred == 0) & (all_labels == 1)).sum()
    # TN = ((all_pred == 0) & (all_labels == 0)).sum()
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    F1 = (2 * precision * recall) / (precision + recall)
    return F1