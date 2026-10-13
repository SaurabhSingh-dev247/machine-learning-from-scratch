from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
import numpy as np
import pandas as pd
from utils import (
    calculate_log_loss,
    calculate_recall,
    calculate_precision,
    calculate_accuracy,
    confusion_matrix,
)

df = pd.read_csv("./data/loan_status.csv")
df["previous_loan_defaults_on_file"] = df["previous_loan_defaults_on_file"].map(
    {"Yes": 1, "No": 0}
)
df["person_gender"] = df["person_gender"].map({"female": 0, "male": 1})
df["person_education"] = df["person_education"].map(
    {"High School": 0, "Associate": 1, "Bachelor": 2, "Master": 3, "Doctorate": 4}
)
df = pd.get_dummies(df, columns=["person_home_ownership", "loan_intent"], dtype=int)

y = df["loan_status"].to_numpy()
x = df.drop(columns=["loan_status"]).to_numpy()

N_train = int((90 / 100) * len(x))

rng = np.random.default_rng(42)

indices = np.arange(len(x))
rng.shuffle(indices)

train_indices = indices[:N_train]
test_indices = indices[N_train:]

x_train = x[train_indices]
x_test = x[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]


def add_bias(X):
    return np.hstack([np.ones((X.shape[0], 1)), X])


x_train_mean = np.mean(x_train, axis=0)
x_train_std_deviation = np.std(x_train, axis=0)

# prevent division by zero
x_train_std_deviation[x_train_std_deviation == 0] = 1

x_train_stdized = (x_train - x_train_mean) / x_train_std_deviation
x_train_stdized = add_bias(x_train_stdized)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def gradient_ascent(Y, X, theta, alpha):
    z = X @ theta
    y_pred = sigmoid(z)
    error = Y - y_pred
    gradient = (X.T @ error) / len(X)
    theta += alpha * gradient
    return theta


def train(Y, X, theta, alpha, epochs):
    for epoch in range(epochs):
        theta = gradient_ascent(Y, X, theta, alpha)
        if epoch % 500 == 0:
            probabilities = sigmoid(X @ theta)
            loss = calculate_log_loss(Y, probabilities)
            print("Epoch:", epoch, "Loss:", loss)

    return theta


theta_initial = np.zeros(shape=x_train_stdized.shape[1], dtype=float)

theta = train(y_train, x_train_stdized, theta_initial, 0.1, 21000)


x_test_scaled = (x_test - x_train_mean) / x_train_std_deviation
x_test_scaled = add_bias(x_test_scaled)

z_new = x_test_scaled @ theta
y_pred_scaled = sigmoid(z_new)


for threshold in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    y_pred = np.where(y_pred_scaled >= threshold, 1, 0)
    TP, TN, FP, FN = confusion_matrix(y_test, y_pred=y_pred)
    print(
        "Threshold:",
        threshold,
        "Accuracy:",
        calculate_accuracy(y_pred, y_test),
        "Precision:",
        calculate_precision(y_pred, y_test),
        "Recall:",
        calculate_recall(y_pred, y_test),
        "Loss:",
        calculate_log_loss(y_test, y_pred_scaled),
    )
    print("CONFUSTION MATRIX:")
    print("TP:", TP, "TN:", TN, "FP:", FP, "FN:", FN)
    print()


clf = LogisticRegression(max_iter=21000, penalty=None)
clf.fit(x_train_stdized[:, 1:], y_train)  # drop the ones column

print("THETA:", theta)
print("SCIKIT THETA:", clf.coef_)
print("SCIKIT INTERCEPT:", clf.intercept_)
print(
    "SCIKIT LEARN LOG LOSS:",
    log_loss(y_test, clf.predict_proba(x_test_scaled[:, 1:])[:, 1]),
    "MY LOG LOSS:",
    calculate_log_loss(y_test, y_pred_scaled),
)

print()
print("PROBABILITY COMPARISON")
my_prob = sigmoid(x_test_scaled @ theta)
sk_prob = clf.predict_proba(x_test_scaled[:, 1:])[:, 1]
print(np.mean(np.abs(my_prob - sk_prob)))
print()
print("PREDICTION COMPARISON")
my_pred = (my_prob >= 0.5).astype(int)
sk_pred = (sk_prob >= 0.5).astype(int)
print(np.mean(my_pred == sk_pred))
