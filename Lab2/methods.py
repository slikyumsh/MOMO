import math
import numpy as np


def f(x):
    return (x[0] ** 2 + (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 2 + x[2] ** 2) / 2 - x[0]


def finite_difference(x):
    h1 = np.array([1, 0, 0])
    h2 = np.array([0, 1, 0])
    h3 = np.array([0, 0, 1])
    # h = np.array([3, 3, 3])
    # return (f(x + h) - f(x)) / h

    tmp = lambda x, h: (f(x + h) - f(x - h)) / (2 * h.max())

    return np.array([tmp(x, h1), tmp(x, h2), tmp(x, h3)])


def finite_difference2(x, i, j):
    l = (f(x + i + j) - f(x + i)) / i.max()
    r = (f(x - i + j) - f(x - i)) / i.max()
    return (l - r) / j.max()


def g(x):
    return np.array([2 * x[0] - x[1] - 1, -x[0] + 2 * x[1] - x[2], 2 * x[2] - x[1]])


def numerical_g(x):
    return np.array(finite_difference(x))


def bfgs(f, grad, initial_gesse, x, epsilon, c1, c2, a_max, max_iter, search_max_iter):
    I = np.eye(3)
    H = initial_gesse
    for iter in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < epsilon:
            break
        p = -H @ g
        a = linear_step_size_search(f, grad, x, p, c1, c2, a_max, search_max_iter)
        s = a * p
        new_x = x + s
        new_g = grad(new_x)
        y = new_g - g

        # y.transpose() here has no effect ; (3,) -> (3,)
        q = (1. / (y.transpose() @ s))
        # Might be error. Exists variation where second np.outer must be (y,s) instead of what in formula in lab2
        H = (I - q * np.outer(s, y)) @ H @ (I - q * np.outer(s, y)) + q * np.outer(s, s)
        x = new_x

    return x, iter


def lbfgs(grad, initial_gesse, x, epsilon, max_iter, mem_limit):
    I = np.eye(3)
    a = []
    s = []
    y = []
    H = initial_gesse
    for iter in range(max_iter):
        g = grad(x)
        if np.linalg.norm(g) < epsilon:
            break

        q = grad(x)
        for i in range(len(s) - 1, -1, -1):
            a[i] = (1 / (y[i] @ s[i])) * s[i] @ q
            q -= a[i] * y[i]

        if iter != 0:
            approx_y = (s[-1] @ y[-1]) / (y[-1] @ y[-1])
            H = approx_y * I
        z = H @ q

        for i in range(len(s)):
            B = (1 / (y[i] @ s[i])) * y[i] @ z
            z = z + s[i] * (a[i] - B)

        z = -z
        new_x = x + z
        if len(s) == mem_limit:
            s.pop(0)
            y.pop(0)
            a.pop(0)

        s.append(new_x - x)
        y.append(grad(new_x) - g)
        a.append(0)

        x = new_x
    return x, iter


def linear_step_size_search(f, grad, x, p, c1, c2, a_max, max_iter):
    a = [0, (0 + a_max) / 2]
    for i in range(1, max_iter):
        approx_f_value = f(x + a[i] * p)
        if approx_f_value > f(x) + c1 * a[i] * grad(x).transpose() @ p or (approx_f_value >= f(x + a[i - 1] * p) and i > 1):
            return zoom(f, grad, x, p, a[i - 1], a[i], c1, c2)
        approx_grad_value = grad(x + a[i] * p)
        if np.linalg.norm(approx_grad_value) <= -c2 * np.linalg.norm(grad(x)):
            return a[i]
        if approx_grad_value.transpose() @ p >= 0:
            return zoom(f, grad, x, p, a[i - 1], a[i], c1, c2)
        a.append((a_max - a[i]) / 2 + a[i])
    return a[-1]


def zoom(f, grad, x, p, a_low, a_high, c1, c2):
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

