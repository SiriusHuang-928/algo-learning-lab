import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def calc_metrics(labels, scores):
    labels = np.asarray(labels)
    scores = np.asarray(scores)

    labels = labels.reshape(-1)
    scores = scores.reshape(-1)

    if len(np.unique(labels)) < 2:
        return 0.0, 0.0, 0.0

    roc_auc = roc_auc_score(labels, scores)
    pr_auc = average_precision_score(labels, scores)

    preds = (scores >= 0.5).astype(int)
    tp = np.sum((preds == 1) & (labels == 1))
    fp = np.sum((preds == 1) & (labels == 0))
    fn = np.sum((preds == 0) & (labels == 1))

    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    pa_f1 = 2 * precision * recall / (precision + recall + 1e-8)

    return roc_auc, pr_auc, pa_f1
