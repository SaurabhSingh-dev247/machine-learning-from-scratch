import numpy as np
from model_asses import accuracy, confusion_matrix, precision, recall
from utils import grad_sigmoid, initialization_limit, sigmoid, softmax

rng = np.random.default_rng(42)


class NN:
    def __init__(self, epochs: int, size: list, lr=1e-1, batch_size=120):
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
        for i in range(0, N, self.batch_size):
            _, DW, DB = self.back_prop(
                X[i : i + self.batch_size], Y[i : i + self.batch_size]
            )

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
