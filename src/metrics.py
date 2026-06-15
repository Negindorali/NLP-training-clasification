def accuracy(y_true, y_pred):

    correct = 0

    for t, p in zip(y_true, y_pred):

        if t == p:
            correct += 1

    return correct / len(y_true)


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
