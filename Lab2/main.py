import math
import numpy as np


def f(x):
    return (x[0] ** 2 + (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 2 + x[2] ** 2) / 2 - x[0]


def grad(x):
    return np.array(
        (2 * x[0] - x[1] - 1, -x[0] + 2 * x[1] - x[2], 2 * x[2] - x[1]),
        dtype=np.float32,
    )


def gesse():
    return np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=np.float32)


def inverse_gesse():
    return np.array(
        [[0.75, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.75]], dtype=np.float32
    )


def bfgs(x=np.array([1, 1, 1]), epsilon=0.1, c1=0.0001, c2=0.9, a_max=1, max_iter=10):
    i = 0
    while np.linalg.norm(grad(x)) > epsilon:
        p = -inverse_gesse().dot(grad(x))
        a = linear_step_size_search(f, grad, x, p, c1, c2, a_max, max_iter)
        x = x + a * p
        i += 1
        print(f"x={x} i={i}")
    return x


def linear_step_size_search(f, grad, x, p, c1, c2, a_max, max_iter):
    a = [0, (0 + a_max) / 2]
    for i in range(1, max_iter):
        approx_f_value = f(x + a[i] * p)
        if approx_f_value > f(x) + c1 * a[i] * grad(x).transpose().dot(p) or (
            approx_f_value >= f(x + a[i - 1] * p) and i > 1
        ):
            return zoom(f, grad, x, p, a[i - 1], a[i], c1, c2)
        approx_grad_value = grad(x + a[i] * p)
        if np.linalg.norm(approx_grad_value) <= -c2 * np.linalg.norm(grad(x)):
            return a[i]
        if approx_grad_value.transpose().dot(p) >= 0:
            return zoom(f, grad, x, p, a[i - 1], a[i], c1, c2)
        a.append((a[i] + a_max) / 2)
    return a[-1]


def zoom(f, grad, x, p, a_low, a_high, c1, c2):
    while True:
        a_j = (a_low + a_high) / 2
        approx_f_value = f(x + a_j * p)
        approx_grad_f_x = grad(x)
        if approx_f_value > f(x) + c1 * a_j * approx_grad_f_x.transpose().dot(p) or (
            approx_f_value >= f(x + a_low * p)
        ):
            a_high = a_j
        else:
            approx_grad = grad(x + a_j * p)
            if np.linalg.norm(approx_grad) <= -c2 * approx_grad_f_x.transpose(p):
                return a_j
            if approx_grad.transpose().dot(p) * (a_high - a_low) >= 0:
                a_high = a_low
            a_low = a_j


x = bfgs()
print(x)
print("huray!")
