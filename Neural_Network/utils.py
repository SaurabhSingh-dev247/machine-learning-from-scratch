import numpy as np
# from main import forward_propagation


def evaluate_mse(forward_prop,w1, b1, w2, b2, x, y):
    predictions = []
    for i in range(len(x)):
        y_pred, *_ = forward_prop(w1, b1, w2, b2, x[i])
        predictions.append(float(y_pred))

    predictions = np.array(predictions)
    return np.mean((predictions - y) ** 2)


def calculate_mse(a):
    return np.mean(a**2)


# def calculate_mae(a):
#     return np.mean(np.abs(a))


def calculate_R2(y, y_pred):
    res_sum = np.sum((y_pred - y) ** 2)
    total_sum = np.sum((y - y.mean()) ** 2)

    return 1 - (res_sum / total_sum)


def stdized_x(x, x_train):
    return (x - np.mean(x_train, axis=0, dtype=float)) / np.std(
        x_train, axis=0, dtype=float
    )


def stdized_y(y, y_train):
    return (y - np.mean(y_train, axis=0, dtype=float)) / np.std(
        y_train, axis=0, dtype=float
    )


# def unscale_X(scaled_X, x_train):
#     return scaled_X * np.std(x_train, axis=0, dtype=float) + np.mean(
#         x_train, axis=0, dtype=float
#     )


def unscale_Y(scaled_Y, y_train):
    return scaled_Y * np.std(y_train, axis=0, dtype=float) + np.mean(
        y_train, axis=0, dtype=float
    )


def leaky_relu(z, ls):

    return np.where(z > 0, z, ls * z)


def grad_leaky_relu(z, ls):
    return np.where(z > 0, 1, ls)
