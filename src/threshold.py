import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, recall_score, precision_score

def find_best_threshold(y_test, probs):
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)

    # Evita que lengths no coincidan
    thresholds = np.append(thresholds, 1)

    # F1 por threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)

    best_f1_idx = np.argmax(f1_scores)
    best_f1_threshold = thresholds[best_f1_idx]

    best_recall_idx = np.argmax(recalls)
    best_recall_threshold = thresholds[best_recall_idx]

    best_precision_idx = np.argmax(precisions)
    best_precision_threshold = thresholds[best_precision_idx]

    return {
        "best_f1_threshold": float(best_f1_threshold),
        "best_recall_threshold": float(best_recall_threshold),
        "best_precision_threshold": float(best_precision_threshold),
        "best_f1_score": float(f1_scores[best_f1_idx]),
        "max_recall": float(recalls[best_recall_idx]),
        "max_precision": float(precisions[best_precision_idx])
    }
