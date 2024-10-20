import math


def f(x):
    return (x[0] ** 2 + (x[0] - x[1]) ** 2 + (x[1] - x[2]) ** 2 + x[2] ** 2) // 2 - x[0]


def linear_step_size_search(f, grad, approx_x, p, c1, c2, a_max, max_iter):
    a = (0, (0 + a_max) / 2)
    for i in range(1, max_iter):
        approx_f_value = f(approx_x + a[i] * p)
        if (
            approx_f_value > f(approx_x) + c1 * a[i] * grad(approx_x)
            or approx_f_value >= f(approx_x + a[i - 1] * p)
            and i > 1
        ):
            return zoom(f, grad, approx_x, a[i - 1], a[i], c1, c2)
        approx_grad_value = grad(approx_x + a[i] * p)
        if math.abs(approx_grad_value) <= -c2 * math.abs(grad(approx_x)):
            return a[i]
        if approx_grad_value >= 0:
            return zoom(f, grad, approx_x, a[i - 1], a[i], c1, c2)
        a.append((a[i] + a_max) / 2)


def zoom(f, grad, approx_x, p, a_low, a_high, c1, c2):
    while True:
        a_j = (a_low + a_high) / 2
        approx_f_value = f(approx_x + a_j * p)
        approx_grad_f_x = grad(approx_x)
        if (approx_f_value > f(approx_x) + c1 * a_j * approx_grad_f_x) or (
            approx_f_value >= f(approx_x + a_low * p)
        ):
            a_high = a_j
        else:
            approx_grad = grad(approx_x + a_j * p)
            if math.abs(approx_grad) <= -c2 * approx_grad_f_x * p:
                return a_j
            if approx_grad * (a_high - a_low) >= 0:
                a_high = a_low
            a_low = a_j
