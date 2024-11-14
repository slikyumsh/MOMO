import numpy as np
import time

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]

    y_true_one_hot = np.zeros_like(y_pred)
    y_true_one_hot[np.arange(m), y_true] = 1

    loss = -np.sum(y_true_one_hot * np.log(y_pred + 1e-15)) / m
    return loss


def numerical_gradient(model, X, y, epsilon=1e-5):
    W = model.W
    b = model.b
    grad_W = np.zeros_like(W)
    grad_b = np.zeros_like(b)

    probs = model.softmax(model.forward(X))
    loss = cross_entropy_loss(y, probs)


    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_eps = W.copy()
            W_eps[i, j] += epsilon
            model.W = W_eps
            probs_eps = model.softmax(model.forward(X))
            loss_eps = cross_entropy_loss(y, probs_eps)
            grad_W[i, j] = (loss_eps - loss) / epsilon
    model.W = W


    for j in range(b.shape[1]):
        b_eps = b.copy()
        b_eps[0, j] += epsilon
        model.b = b_eps
        probs_eps = model.softmax(model.forward(X))
        loss_eps = cross_entropy_loss(y, probs_eps)
        grad_b[0, j] = (loss_eps - loss) / epsilon
    model.b = b

    return grad_W, grad_b



def bfgs_optimizer(model, X_train, y_train, num_iters=100):
    n_params = model.W.size + model.b.size
    xk = np.concatenate([model.W.flatten(), model.b.flatten()])
    Hk = np.eye(n_params)
    epsilon = 1e-5

    start_time = time.time()

    for _ in range(1, num_iters + 1):
        grad_W, grad_b = numerical_gradient(model, X_train, y_train, epsilon)
        grad = np.concatenate([grad_W.flatten(), grad_b.flatten()])

        if np.linalg.norm(grad) < epsilon:
            break

        pk = -Hk @ grad

        alpha = 1e-3
        xk_new = xk + alpha * pk

        sk = xk_new - xk
        xk = xk_new

        model.W = xk[:model.W.size].reshape(model.W.shape)
        model.b = xk[model.W.size:].reshape(model.b.shape)

        grad_W_new, grad_b_new = numerical_gradient(model, X_train, y_train, epsilon)
        grad_new = np.concatenate([grad_W_new.flatten(), grad_b_new.flatten()])

        yk = grad_new - grad
        rho_k = 1.0 / (yk @ sk)
        I = np.eye(n_params)
        Hk = (I - rho_k * np.outer(sk, yk)) @ Hk @ (I - rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)

        grad = grad_new

    total_time = time.time() - start_time
    probs_train = model.softmax(model.forward(X_train))
    final_loss = cross_entropy_loss(y_train, probs_train)

    return final_loss, total_time


def lbfgs_optimizer(model, X_train, y_train, num_iters=100, m=10):
    n_params = model.W.size + model.b.size
    xk = np.concatenate([model.W.flatten(), model.b.flatten()])

    epsilon = 1e-5
    grad_W, grad_b = numerical_gradient(model, X_train, y_train, epsilon)
    gk = np.concatenate([grad_W.flatten(), grad_b.flatten()])

    s_list = []
    y_list = []

    start_time = time.time()

    for _ in range(1, num_iters + 1):
        if np.linalg.norm(gk) < epsilon:
            break

        q = gk.copy()
        alpha_list = []
        rho_list = []

        for i in range(len(s_list)-1, -1, -1):
            s_i = s_list[i]
            y_i = y_list[i]
            rho_i = 1.0 / (y_i @ s_i)
            rho_list.append(rho_i)
            alpha_i = rho_i * (s_i @ q)
            alpha_list.append(alpha_i)
            q = q - alpha_i * y_i

        if len(s_list) > 0:
            gamma_k = (s_list[-1] @ y_list[-1]) / (y_list[-1] @ y_list[-1])
            Hk0 = gamma_k * np.eye(n_params)
        else:
            Hk0 = np.eye(n_params)

        r = Hk0 @ q

        for i in range(len(s_list)):
            s_i = s_list[i]
            y_i = y_list[i]
            rho_i = rho_list[-(i+1)]
            alpha_i = alpha_list[-(i+1)]
            beta_i = rho_i * (y_i @ r)
            r = r + s_i * (alpha_i - beta_i)

        pk = -r

        alpha = 1e-3
        xk_new = xk + alpha * pk
        sk = xk_new - xk
        xk = xk_new

        model.W = xk[:model.W.size].reshape(model.W.shape)
        model.b = xk[model.W.size:].reshape(model.b.shape)

        grad_W_new, grad_b_new = numerical_gradient(model, X_train, y_train, epsilon)
        gk_new = np.concatenate([grad_W_new.flatten(), grad_b_new.flatten()])
        yk = gk_new - gk

        s_list.append(sk)
        y_list.append(yk)

        if len(s_list) > m:
            s_list.pop(0)
            y_list.pop(0)

        gk = gk_new

    total_time = time.time() - start_time
    probs_train = model.softmax(model.forward(X_train))
    final_loss = cross_entropy_loss(y_train, probs_train)

    return final_loss, total_time


def adam_optimizer(model, X_train, y_train, num_iters=100, learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8):
    n_params = model.W.size + model.b.size
    xk = np.concatenate([model.W.flatten(), model.b.flatten()])

    m_t = np.zeros(n_params)
    v_t = np.zeros(n_params)

    start_time = time.time()

    for iter in range(1, num_iters + 1):
        grad_W, grad_b = numerical_gradient(model, X_train, y_train)
        grad = np.concatenate([grad_W.flatten(), grad_b.flatten()])

        m_t = beta1 * m_t + (1 - beta1) * grad
        v_t = beta2 * v_t + (1 - beta2) * (grad ** 2)

        m_hat = m_t / (1 - beta1 ** iter)
        v_hat = v_t / (1 - beta2 ** iter)

        xk = xk - learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        model.W = xk[:model.W.size].reshape(model.W.shape)
        model.b = xk[model.W.size:].reshape(model.b.shape)

    total_time = time.time() - start_time
    probs_train = model.softmax(model.forward(X_train))
    final_loss = cross_entropy_loss(y_train, probs_train)

    return final_loss, total_time
