from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, root_mean_squared_error, r2_score
import numpy as np
import pandas as pd

# loading the data
df = pd.read_csv("./data/ds2.csv")
encoding = {"yes": 1, "no": 0}

df[
    [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
    ]
] = df[
    [
        "mainroad",
        "guestroom",
        "basement",
        "hotwaterheating",
        "airconditioning",
        "prefarea",
    ]
].replace(encoding)

df = pd.get_dummies(df, columns=["furnishingstatus"], dtype= int)

y = df["price"].to_numpy()
x = df.drop(columns=["price"]).to_numpy()

N_train = int((90/100) * len(x))

print("N TRAIN:", N_train)

rng = np.random.default_rng(42)

indices = np.arange(len(x))
rng.shuffle(indices)

train_indices = indices[:N_train]
test_indices = indices[N_train:]

x_train = x[train_indices]
x_test = x[test_indices]

y_train = y[train_indices]
y_test = y[test_indices]

x_train_mean = np.mean(x_train, axis=0)
x_std_deviation = np.std(x_train, axis=0, dtype=float)
x_std_deviation[x_std_deviation == 0] = 1
x_train_scaled = (x_train - x_train_mean) / x_std_deviation

x_test_mean = np.mean(x_test)
x_test_std_deviation = np.std(x_test)
x_test_scaled = (x_test - x_test_mean) / x_test_std_deviation

y_mean = np.mean(y_train)
y_std_deviation = np.std(y_train)
if y_std_deviation == 0:
    y_std_deviation = 1

y_train_scaled = (y_train - y_mean) / y_std_deviation


def predict(x, theta):
    return x @ theta


def gradient_descent(X, y, theta, alpha):
    N = len(y)
    err = X @ theta - y
    grad = (1 / N) * (X.T @ err)
    return theta - alpha * grad


def train(y, x, theta, alpha, epochs):
    for i in range(epochs):
        theta = gradient_descent(x, y, theta, alpha)
        avg_loss = calculate_average_loss(
            y,
            x,
            theta,
        )

        if i % 500 == 0:
            print("EPOCH:",i)
            print("Average loss:", avg_loss)
    return theta


def calculate_average_loss(y, x, theta):
    error = (x @ theta) - y
    return 0.5 * np.mean(error**2)


x_train_scaled = np.hstack([np.ones((x_train.shape[0], 1)), x_train_scaled])
x_test_scaled = np.hstack([np.ones((x_test.shape[0], 1)), x_test_scaled])

theta_initial = np.zeros(x_train_scaled.shape[1], dtype=float)

theta = train(y_train_scaled, x_train_scaled, theta_initial, 0.1, 2000)

model = LinearRegression(fit_intercept=True, n_jobs=2500)
model.fit(X=x_train_scaled[:, 1:], y=y_train_scaled)

coeff = model.coef_
intercept = model.intercept_

print("THETA AFTER TRAINING:", theta[1:])
print("INTERCEPT:", theta[0])
print("SCIKIT THETA AFTER TRAINING:", coeff)
print("SCIKIT INTERCEPT:", intercept)
print()

y_pred_scaled = predict(x_test_scaled, theta)
y_pred_scaled_scikit = model.predict(x_test_scaled[:, 1:])
y_pred = y_pred_scaled * y_std_deviation + y_mean
y_pred_scikit =  y_pred_scaled_scikit * y_std_deviation + y_mean


rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
mae = np.mean(np.abs(y_pred - y_test))
r2 = 1 - np.sum((y_test - y_pred) ** 2) / np.sum((y_test - y_test.mean()) ** 2)

print(f"RMSE: {rmse} V/S SCIKIT RMSE {root_mean_squared_error(y_test, y_pred_scikit)}")
print(f"MAE: {mae} V/S SCIKIT MAE {mean_absolute_error(y_test, y_pred_scikit)}")
print(f"R2: {r2} V/S SCIKIT R2 {r2_score(y_test, y_pred_scikit)}")
