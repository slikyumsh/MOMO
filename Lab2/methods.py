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


def adam(grad, x, max_iter, learning_rate=0.1, beta1=0.9, beta2=0.999, eps=1e-8):
    s = np.zeros(3)
    v = np.zeros(3)

    for t in range(max_iter):
        g = grad(x)
        s = beta1 * s + (1. - beta1) * g
        v = beta2 * v + (1. - beta2) * g ** 2
        x = x - learning_rate * s / (1. - beta1 ** (t + 1)) / (np.sqrt(v / (1. - beta2 ** (t + 1))) + eps)
    return x, max_iter


x = np.array([5, 5, 5])
gesse = np.array([
    [
        finite_difference2(x, np.array((1, 0, 0)), np.array((1, 0, 0))),
        finite_difference2(x, np.array((1, 0, 0)), np.array((0, 1, 0))),
        finite_difference2(x, np.array((1, 0, 0)), np.array((0, 0, 1)))
    ],
    [
        finite_difference2(x, np.array((0, 1, 0)), np.array((1, 0, 0))),
        finite_difference2(x, np.array((0, 1, 0)), np.array((0, 1, 0))),
        finite_difference2(x, np.array((0, 1, 0)), np.array((0, 0, 1)))
    ],
    [
        finite_difference2(x, np.array((0, 0, 1)), np.array((1, 0, 0))),
        finite_difference2(x, np.array((0, 0, 1)), np.array((0, 1, 0))),
        finite_difference2(x, np.array((0, 0, 1)), np.array((0, 0, 1)))
    ]
])

epsilon = 1e-4
c1 = 0.0001
c2 = 0.9
a_max = 1
max_iter = 10000
search_max_iter = 100
mem_limit = 10
initial_gesse = np.eye(3)
analitic_gesse = np.array([
    [0.75, 0.5, 0.25],
    [0.5, 1, 0.5],
    [0.25, 0.5, 0.75]
])

print("Task 1.1")
print("Алгоритмы BFGS и L-BFGS применимы к этой задаче, так как функция f(x) является дважды дифференцируемой выпуклой функцией.")
print()
print("Task 1.2")

point, iter = bfgs(f, g, initial_gesse, x, epsilon, c1, c2, a_max, max_iter, search_max_iter)
print(f'BFGS (a) x={point}, f(x)={f(point)} iter={iter}')

point, iter = bfgs(f, numerical_g, initial_gesse, x, epsilon, c1, c2, a_max, max_iter, search_max_iter)
print(f'BFGS (b) x={point}, f(x)={f(point)} iter={iter}')

point, iter = lbfgs(g, initial_gesse, x, epsilon, max_iter, mem_limit)
print(f'L-BFGS (a) x={point}, f(x)={f(point)} iter={iter}')

point, iter = lbfgs(numerical_g, initial_gesse, x, epsilon, max_iter, mem_limit)
print(f'L-BFGS (b) x={point}, f(x)={f(point)} iter={iter}')
print()
print("Task 1.3")
print("Approximate Gesse matrix")
print(gesse)
point, iter = bfgs(f, g, gesse, x, epsilon, c1, c2, a_max, max_iter, search_max_iter)
print(f'BFGS x={point}, f(x)={f(point)} iter={iter}')
print()
print("Analitic Gesse matrix")
print(analitic_gesse)
point, iter = bfgs(f, g, analitic_gesse, x, epsilon, c1, c2, a_max, max_iter, search_max_iter)
print(f'BFGS x={point}, f(x)={f(point)} iter={iter}')
print()
point, iter = adam(numerical_g, x, max_iter)
print(f'ADAM x={point}, f(x)={f(point)} iter={iter}')
print()
