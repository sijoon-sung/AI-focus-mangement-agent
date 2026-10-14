import numpy as np


def classification_metrics(labels, probabilities):
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    pred = p.argmax(axis=1)
    cm = np.zeros((4, 4), dtype=int)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm)
    recall = tp / np.maximum(cm.sum(1), 1)
    precision = tp / np.maximum(cm.sum(0), 1)
    f1 = 2 * precision * recall / np.maximum(precision + recall, 1e-12)
    confidence = p.max(1)
    ece = 0.
    for lo, hi in zip(np.linspace(0, 1, 16)[:-1], np.linspace(0, 1, 16)[1:]):
        mask = (confidence >= lo) & ((confidence < hi) if hi < 1 else (confidence <= hi))
        if mask.any():
            ece += mask.mean() * abs((pred[mask] == y[mask]).mean() - confidence[mask].mean())
    support = cm.sum(1)
    return {'accuracy': float((pred == y).mean()), 'macro_f1': float(f1.mean()),
            'balanced_accuracy': float(recall[support > 0].mean()),
            'ordinal_mae': float(np.abs(pred - y).mean()), 'ece_15_bins': float(ece),
            'nll': float(-np.log(p[np.arange(len(y)), y].clip(1e-12)).mean()),
            'per_class_recall': recall.tolist(), 'support': support.tolist(),
            'confusion_matrix': cm.tolist()}
