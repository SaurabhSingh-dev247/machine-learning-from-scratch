import numpy as np


def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))


def softmax(z):
    z = z - np.max(z, axis=0, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


def grad_sigmoid(Z):
    a = sigmoid(Z)
    dZ = a * (1 - a)
    return dZ


def scale_train_data(Z):
    std_deviation = np.std(Z, axis=0, dtype=float)
    std_deviation[std_deviation == 0] = 1
    return (Z - np.mean(Z, axis=0, dtype=float)) / std_deviation


def scale_test_data(Z, Z_train):
    std_deviation = np.std(Z_train, axis=0, dtype=float)
    std_deviation[std_deviation == 0] = 1
    return (Z - np.mean(Z_train, axis=0, dtype=float)) / std_deviation

def initialization_limit(f_in, f_out):
    return np.sqrt(6 / (f_in + f_out))