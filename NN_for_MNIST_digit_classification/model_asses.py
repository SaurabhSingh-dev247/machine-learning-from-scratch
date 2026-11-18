import numpy as np


def encode_predicitons(preds):
    max_indexes = np.argmax(preds, axis=1)
    new_arr = np.zeros(shape=preds.shape, dtype=int)
    new_arr[np.arange(preds.shape[0]), max_indexes] = 1
    return new_arr


def confusion_matrix(preds, labels):
    conf_matrix = np.zeros((10, 10), dtype=int)

    predicted_classes = np.argmax(preds, axis=1)
    actual_classes = np.argmax(labels, axis=1)

    np.add.at(conf_matrix, (actual_classes, predicted_classes), 1)

    return conf_matrix


def accuracy(conf_matrix):
    return np.sum(np.diagonal(conf_matrix)) / np.sum(np.sum(conf_matrix, axis=1))


def precision(conf_matrix):
    cm = np.asarray(conf_matrix, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {cm.shape}")

    tp = np.diag(cm)
    predicted = cm.sum(axis=0)

    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)

    return np.mean(precision)


def recall(conf_matrix):
    cm = np.asarray(conf_matrix, dtype=float)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError(f"Expected a square matrix, got shape {cm.shape}")

    cm = np.asarray(conf_matrix, dtype=float)
    tp = np.diag(cm)

    actual = cm.sum(axis=1)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)

    return np.mean(recall)
