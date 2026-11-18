import numpy as np  # noqa: I001
import pandas as pd
from utils import (
    grad_sigmoid,
    scale_test_data,
    scale_train_data,
    sigmoid,
    softmax,
    initialization_limit,
)
from model_asses import (
    encode_predicitons,
    confusion_matrix,
    accuracy,
    precision,
    recall,
)

train_df = pd.read_csv("./data/mnist_train.csv")
test_df = pd.read_csv("./data/mnist_test.csv")

# labels_count = pd.unique(test_df["label"]).sum()

y_train = train_df["label"]
y_test = test_df["label"]
test_indices = np.arange(len(y_test))

y_train = pd.get_dummies(y_train, dtype=int).to_numpy()
y_test = pd.get_dummies(y_test, dtype=int).to_numpy()

rng = np.random.default_rng(42)
rng.shuffle(test_indices)

x_train = train_df.drop(columns=["label"]).to_numpy()
x_test = test_df.drop(columns=["label"]).to_numpy()

x_train_scaled = scale_train_data(x_train)
x_test_scaled = scale_test_data(x_test, x_train)

features = x_train.shape[1]
layers = [features, 28, 28, 10]
batch_size = 120


class NN:
    def __init__(self, epochs: int, size: list, lr=1e-1):
        N = len(size)
        self.epochs = epochs
        self.size = size
        self.lr = lr
        self.W = [
            rng.uniform(
                low=-(initialization_limit(size[i], size[i + 1])),
                high=(initialization_limit(size[i], size[i + 1])),
                size=(size[i + 1], size[i]),
            )
            for i in range(N - 1)
        ]
        self.B = [np.zeros(shape=(size[i], 1)) for i in range(1, N)]

    def forward_prop(self, X):
        L = len(self.W)
        A = [X.T]
        Z = []

        for i in range(L):
            z = self.W[i] @ A[i] + self.B[i]
            Z.append(z)
            if i == L - 1:
                A.append(softmax(z))  # output layer
            else:
                A.append(sigmoid(z))  # hidden layers

        pred = A[-1]
        return pred, Z, A

    def compute_loss(self, Y_pred, Y):
        losses = []
        for i in range(len(Y)):
            loss = -np.sum(Y[i] * np.log(Y_pred[i]))
            losses.append(loss)
        return np.mean(losses)

    def back_prop(self, X, Y):
        pred, Z, A = self.forward_prop(X=X)

        L = len(self.W)
        m = X.shape[0]
        DW = [None] * L
        DB = [None] * L

        err_sig = pred - Y.T  # (10, m)
        DW[L - 1] = err_sig @ A[L - 1].T / m
        DB[L - 1] = np.sum(err_sig, axis=1, keepdims=True) / m

        for i in range(L - 2, -1, -1):
            err_sig = (self.W[i + 1].T @ err_sig) * grad_sigmoid(Z=Z[i])
            DW[i] = err_sig @ A[i].T / m
            DB[i] = np.sum(err_sig, axis=1, keepdims=True) / m

        return pred, DW, DB

    def update_params(self, X, Y):
        N = len(X)
        for i in range(0, N, batch_size):
            _, DW, DB = self.back_prop(X[i : i + batch_size], Y[i : i + batch_size])

            for j in range(len(DW)):
                self.W[j] -= self.lr * DW[j]
                self.B[j] -= self.lr * DB[j]


    def train(self, X, Y):
        indices = np.arange(len(X))
        for i in range(1, self.epochs + 1):
            rng.shuffle(indices)
            X_shuffled = X[indices]
            Y_shuffled = Y[indices]

            self.update_params(X_shuffled, Y_shuffled)
            Y_pred_all, _, _ = self.forward_prop(X=X_shuffled)
            loss = self.compute_loss(Y_pred_all.T, Y_shuffled)
            conf_matrix = confusion_matrix(preds=Y_pred_all.T, labels=Y_shuffled)

            acc = accuracy(conf_matrix=conf_matrix)
            prec = precision(conf_matrix=conf_matrix)
            rec = recall(conf_matrix=conf_matrix)

            print(
                "Epoch:",
                i,
                "Loss:",
                loss,
                "Accuracy:",
                acc,
                "Precision:",
                prec,
                "Recall:",
                rec,
            )

        return self.W, self.B


model = NN(100, size=layers, lr=3e-2)
W, B = model.train(x_train_scaled, y_train)

x_test_shuffled = x_test_scaled[test_indices]
y_test_shuffled = y_test[test_indices]

y_pred, _, _ = model.forward_prop(x_test_shuffled)
y_pred = np.transpose(y_pred)

y_pred_encoded = encode_predicitons(y_pred)
conf_matrix = confusion_matrix(y_pred_encoded, y_test_shuffled)
accy = accuracy(conf_matrix=conf_matrix)
prec = precision(conf_matrix=conf_matrix)
recl = recall(conf_matrix=conf_matrix)

test_loss = model.compute_loss(y_pred, y_test_shuffled)

print()
print("TEST LOSS:", test_loss)
print("ACCURACY:", accy)
print("PRECISION:", prec)
print("RECALL:", recl)
print("CONFUSION MATRIX:")
print( conf_matrix)