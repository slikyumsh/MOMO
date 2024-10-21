import math
import numpy as np


def f(x):
    return (x[0] ** 2 + (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 2 + x[2] ** 2) / 2 - x[0]


def grad(x):
    return np.array([2 * x[0] - x[1] - 1, -x[0] + 2 * x[1] - x[2], 2 * x[2] - x[1]],dtype=np.float32)


def gesse_test():
    return np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 2]], dtype=np.float32)


def inverse_gesse_test():
    return np.array([[0.75, 0.5, 0.25], [0.5, 1, 0.5], [0.25, 0.5, 0.75]], dtype=np.float32)


def gesse_initial():
    return np.ones((3, 3), dtype=np.float32)


def bfgs(x=np.array([0, 0, 0], dtype=np.float32), epsilon=0.5, c1=0.0001, c2=0.9, a_max=1, max_iter=10):
    i = 0
    s = 0
    y = 0
    H = gesse_initial()
    norm = np.linalg.norm(grad(x))
    while norm > epsilon:
        print(H)
        p = -H.dot(grad(x))
        a = linear_step_size_search(f, grad, x, p, c1, c2, a_max, max_iter)
        x = x + a * p
        s = a * p
        y = grad(x) - grad(x - s)

        precompute = y.transpose().dot(s)
        precompute = 1 if precompute == 0 else precompute

        precompute = 1 / precompute
        precompute = precompute * s

        H = (np.ones((3, 3), dtype=np.float32) - precompute.dot(y.transpose()))\
            .dot(H).dot(np.ones((3, 3), dtype=np.float32) - precompute.dot(y.transpose()))\
            + precompute.dot(s.transpose())
        i += 1
        norm = np.linalg.norm(grad(x))
        print(f"i={i} norm={norm} x={x}")
    return x


def linear_step_size_search(f, grad, x, p, c1, c2, a_max, max_iter):
    a = [0.01, (0.01 + a_max) / 2]
    for i in range(1, max_iter):
        approx_f_value = f(x + a[i] * p)
        if approx_f_value > f(x) + c1 * a[i] * grad(x).transpose().dot(p) or (approx_f_value >= f(x + a[i - 1] * p) and i > 1):
            return zoom(f, grad, x, p, a[i - 1], a[i], c1, c2)
        approx_grad_value = grad(x + a[i] * p)
        if np.linalg.norm(approx_grad_value) <= -c2 * np.linalg.norm(grad(x)):
            return a[i]
        if approx_grad_value.transpose().dot(p) >= 0:
            return zoom(f, grad, x, p, a[i - 1], a[i], c1, c2)
        a.append((a_max - a[i]) / 2 + a[i])
    return a[-1]


def zoom(f, grad, x, p, a_low, a_high, c1, c2):
    while True:
        a_j = (a_low + a_high) / 2
        if f(x + a_j * p) > f(x) + c1 * a_j * grad(x).transpose().dot(p) or f(x + a_j * p) >= f(x + a_low * p):
            a_high = a_j
        else:
            approx_grad = grad(x + a_j * p)
            if np.linalg.norm(approx_grad) <= -c2 * grad(x).transpose().dot(p):
                return a_j
            if approx_grad.transpose().dot(p) * (a_high - a_low) >= 0:
                a_high = a_low
            a_low = a_j

        if math.fabs(a_low - a_high) < 1e-4:
            return a_low

x = bfgs()
print(x)
