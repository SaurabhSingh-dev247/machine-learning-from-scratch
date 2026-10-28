import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from utils import (
    evaluate_mse,
    calculate_R2,
    calculate_mse,
    stdized_x,
    stdized_y,
    unscale_Y,
    leaky_relu,
    grad_leaky_relu,
)

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
].replace(
    encoding
)
df = pd.get_dummies(df, columns=["furnishingstatus"], dtype=int)

y = df["price"].to_numpy()
x = df.drop(columns=["price"]).to_numpy()

N_train = int((70 / 100) * len(x))
lr = 1e-5
ls = 1e-1
hidden_neurons = 4
iters = 3000

rng = np.random.default_rng(42)
indices = np.arange(len(x))
rng.shuffle(indices)

x = x[indices]
y = y[indices]


x_train = x[:N_train]
y_train = y[:N_train]

x_test = x[N_train:]
y_test = y[N_train:]

prediction_losses = []


def predict(w1, b1, w2, b2, x):
    prediction = []
    for i in range(len(x)):
        y, *_ = forward_propagation(w1, b1, w2, b2, x[i])
        prediction.append(float(y))
    return np.array(prediction)


def forward_propagation(w1, b1, w2, b2, x):
    Z1 = w1 @ x.T + b1
    A1 = leaky_relu(Z1, ls=ls)

    y = w2 @ A1 + b2

    return (
        y,
        A1,
        Z1,
    )


def back_prop(w1, b1, w2, b2, x, y):
    y_pred, A1, Z1 = forward_propagation(w1, b1, w2, b2, x)

    error = y_pred - y

    # Output layer
    dw2 = error * A1
    db2 = error

    #  hidden layer
    error_signal_1 = (w2 * error) * grad_leaky_relu(Z1, ls=ls)

    dw1 = np.outer(error_signal_1, x)
    db1 = error_signal_1

    return (
        dw2,
        db2,
        dw1,
        db1,
    )


def update_parameters(w1, b1, w2, b2, x, y):
    for i in range(len(x)):
        (
            dw2,
            db2,
            dw1,
            db1,
        ) = back_prop(w1, b1, w2, b2, x[i], y[i])

        w1 = w1 - lr * dw1
        b1 = b1 - lr * db1

        w2 = w2 - lr * dw2
        b2 = b2 - lr * db2

    return w1, w2, b1, b2


def train(w1, b1, w2, b2, x, y, epochs: int):
    for i in range(epochs):
        (
            w1,
            w2,
            b1,
            b2,
        ) = update_parameters(w1, b1, w2, b2, x, y)
        # errors = np.array(losses)

        if i % 100 == 0:
            train_mse = evaluate_mse(forward_propagation, w1, b1, w2, b2, x, y)
            print("Epoch:", i)
            print("Training MSE:", train_mse)
            # print("MAE:", calculate_mae(errors))

    return w1, w2, b1, b2


features = x.shape[1]

w1_init = rng.normal(0, np.sqrt(2 / features), size=(hidden_neurons, features))

w2_init = rng.normal(0, np.sqrt(2 / hidden_neurons), size=(hidden_neurons,))

b1_init = np.zeros((hidden_neurons,), dtype=float)
b2_init = 0.0

x_train_scaled = stdized_x(x_train, x_train=x_train)
y_train_scaled = stdized_y(y_train, y_train=y_train)

x_test_scaled = stdized_x(x_test, x_train=x_train)
y_test_scaled = stdized_y(y_test, y_train=y_train)

w1, w2, b1, b2 = train(
    w1=w1_init,
    b1=b1_init,
    w2=w2_init,
    b2=b2_init,
    x=x_train_scaled,
    y=y_train_scaled,
    epochs=iters,
)

model = MLPRegressor(
    hidden_layer_sizes=(hidden_neurons,),
    activation="relu",
    solver="sgd",
    max_iter=2500,
)

model.fit(X=x_train_scaled, y=y_train_scaled)
skikit_pred = model.predict(X=x_test_scaled)

y_pred_scaled = predict(w1, b1, w2, b2, x=x_test_scaled)

y_pred_real = unscale_Y(y_pred_scaled, y_train=y_train)
y_pred_real_skikit = unscale_Y(skikit_pred, y_train=y_train)

# for i in range(len(y_pred_real_skikit)):
#     print(
#         "Y real:",
#         y_test[i],
#         "",
#         "Scikit prediction:",
#         y_pred_real_skikit[i],
#         "",
#         "Y prediction impl:",
#         y_pred_real[i],
#     )

skikit_r2 = model.score(X=x_test_scaled, y=skikit_pred)
r2 = calculate_R2(y_test_scaled, np.array(y_pred_scaled))

nn_mse = calculate_mse(np.array(y_pred_scaled - y_test_scaled))
skikit_mse = calculate_mse(np.array(skikit_pred - y_test_scaled))

print()
print("Training Finished")
print("R2:", r2)
print("Skikit learn R2:", skikit_r2)
print("Test mse my nn:", nn_mse)
print("Test mse sci-kit nn:", skikit_mse)

plt.title("Mean Squared Error Comparison")
plt.bar(["NN MSE", "SCIKIT NN MSE"], [nn_mse, skikit_mse], width=0.6)
plt.show()
