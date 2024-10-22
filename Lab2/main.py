import math
import numpy as np


def f(x):
    return (x[0] ** 2 + (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 2 + x[2] ** 2) / 2 - x[0]


def grad(x):
    return np.array([2 * x[0] - x[1] - 1, -x[0] + 2 * x[1] - x[2], 2 * x[2] - x[1]])


def bfgs(x=np.array([0, 0, 0]), epsilon=1e-5, c1=0.0001, c2=0.9, a_max=1, max_iter=1000, search_max_iter=100):
    I = np.eye(3)
    H = I
    for _ in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < epsilon:
            return x
        p = -H @ g
        a = linear_step_size_search(x, p, c1, c2, a_max, search_max_iter)
        s = a * p
        new_x = x + s
        new_g = grad(new_x)
        y = new_g - g

        # y.transpose() here has no effect ; (3,) -> (3,)
        q = (1. / (y.transpose() @ s))
        # Might be error. Exists variation where second np.outer must be (y,s) instead of what in formula in lab2
        H = (I - q * np.outer(s, y)) @ H @ (I - q * np.outer(s, y)) + q * np.outer(s, s)
        x = new_x

    return x


def linear_step_size_search(x, p, c1, c2, a_max, max_iter):
    a = [0, (0 + a_max) / 2]
    for i in range(1, max_iter):
        approx_f_value = f(x + a[i] * p)
        if approx_f_value > f(x) + c1 * a[i] * grad(x).transpose() @ p or (approx_f_value >= f(x + a[i - 1] * p) and i > 1):
            return zoom(x, p, a[i - 1], a[i], c1, c2)
        approx_grad_value = grad(x + a[i] * p)
        if np.linalg.norm(approx_grad_value) <= -c2 * np.linalg.norm(grad(x)):
            return a[i]
        if approx_grad_value.transpose() @ p >= 0:
            return zoom(x, p, a[i - 1], a[i], c1, c2)
        a.append((a_max - a[i]) / 2 + a[i])
    return a[-1]


def zoom(x, p, a_low, a_high, c1, c2):
    while True:
        a_j = (a_low + a_high) / 2
        if f(x + a_j * p) > f(x) + c1 * a_j * grad(x).transpose() @ p or f(x + a_j * p) >= f(x + a_low * p):
            a_high = a_j
        else:
            approx_grad = grad(x + a_j * p)
            if np.linalg.norm(approx_grad) <= -c2 * grad(x).transpose() @ p:
                return a_j
            if approx_grad.transpose() @ p * (a_high - a_low) >= 0:
                a_high = a_low
            a_low = a_j

        if math.fabs(a_low - a_high) <= 1e-2:
            return (a_low + a_high) / 2


x = bfgs()
print(f'Funtion min point = {x}; f(x) = {f(x)}')
