def accuracy(y_true, y_pred):

    correct = 0

    for t, p in zip(y_true, y_pred):

        if t == p:
            correct += 1

    return correct / len(y_true)


def precision_recall_f1(y_true, y_pred, labels=None):
    """Per-class precision, recall and F1, plus their macro averages.

    Returns (per_class, macro). per_class maps each label to a dict with
    "precision", "recall", "f1" and "support". macro is a dict with the
    unweighted mean of each metric over the labels.

    labels defaults to the labels present in y_true, so a predicted-only
    class (e.g. the neutral class) is counted as a wrong prediction for
    the true class instead of being scored as a class of its own.
    """

    if labels is None:
        labels = sorted(set(y_true))

    per_class = {}

    for c in labels:

        tp = fp = fn = 0

        for t, p in zip(y_true, y_pred):

            if p == c and t == c:
                tp += 1
            elif p == c and t != c:
                fp += 1
            elif p != c and t == c:
                fn += 1

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }

    macro = {
        "precision": sum(m["precision"] for m in per_class.values()) / len(labels),
        "recall": sum(m["recall"] for m in per_class.values()) / len(labels),
        "f1": sum(m["f1"] for m in per_class.values()) / len(labels),
    }

    return per_class, macro


def confusion_matrix(y_true, y_pred, labels=None):
    """Confusion matrix sized to the labels actually present.

    Returns (matrix, labels). Rows are true labels, columns are predicted
    labels, both in the order of `labels`. Predicted labels that never occur
    in the truth (e.g. the neutral class) still get their own column.
    """

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))

    index = {label: i for i, label in enumerate(labels)}
    size = len(labels)

    matrix = [[0] * size for _ in range(size)]

    for t, p in zip(y_true, y_pred):
        matrix[index[t]][index[p]] += 1

    return matrix, labels
