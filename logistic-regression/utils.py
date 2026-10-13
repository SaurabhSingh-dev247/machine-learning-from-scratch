import numpy as np


def calculate_accuracy(y_predicted, y_actual):
    correct = 0

    for i in range(len(y_actual)):
        if y_predicted[i] == y_actual[i]:
            correct += 1

    return correct / len(y_actual)


def calculate_precision(y_predicted, y_actual):
    TP = 0
    FP = 0
    for i in range(len(y_actual)):
        if (y_predicted[i] == y_actual[i]) and (y_actual[i] == 1):
            TP += 1
        elif (y_predicted[i] == 1) and (y_actual[i] == 0):
            FP += 1
    return TP / (TP + FP)


def calculate_recall(y_predicted, y_actual):
    TP = 0
    FN = 0

    for i in range(len(y_actual)):
        if (y_predicted[i] == y_actual[i]) and (y_actual[i] == 1):
            TP += 1
        elif (y_predicted[i] == 0) and (y_actual[i] == 1):
            FN += 1
    return TP / (TP + FN)



def calculate_log_loss(y, y_hat):
    eps = 1e-15
    y_hat = np.clip(y_hat, eps, 1 - eps)
    T1 = y * np.log(y_hat)
    T2 = (1 - y) * np.log(1 - y_hat)
    return -np.mean(T1 + T2)

def confusion_matrix(y_real, y_pred):
    TP = 0
    FP = 0 
    TN = 0
    FN = 0
    for i in range(len(y_real)):
        if (y_pred[i] == 1) and (y_real[i] == 1):
            TP += 1
        elif  (y_pred[i] == 0) and (y_real[i] == 0):
            TN += 1
        elif  (y_pred[i] == 1) and (y_real[i] == 0):
            FP += 1
        elif  (y_pred[i] == 0) and (y_real[i] == 1):
                FN += 1
    return TP, TN, FP, FN