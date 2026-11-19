import numpy as np  # noqa: I001
import pandas as pd
from nn import NN
from utils import (
    scale_test_data,
    scale_train_data,
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
print(conf_matrix)
