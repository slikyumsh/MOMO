import numpy as np

def cross_entropy_loss(y_true, y_pred):
    m = y_true.shape[0]
    # Создаем one-hot представление меток
    y_true_one_hot = np.zeros_like(y_pred)
    y_true_one_hot[np.arange(m), y_true] = 1
    # Вычисляем потери
    loss = -np.sum(y_true_one_hot * np.log(y_pred + 1e-15)) / m
    return loss


def numerical_gradient(model, X, y, epsilon=1e-5):
    W = model.W
    b = model.b
    grad_W = np.zeros_like(W)
    grad_b = np.zeros_like(b)
    
    # Вычисляем потери при текущих параметрах
    probs = model.softmax(model.forward(X))
    loss = cross_entropy_loss(y, probs)
    
    # Градиент по W
    for i in range(W.shape[0]):
        for j in range(W.shape[1]):
            W_eps = W.copy()
            W_eps[i, j] += epsilon
            model.W = W_eps
            probs_eps = model.softmax(model.forward(X))
            loss_eps = cross_entropy_loss(y, probs_eps)
            grad_W[i, j] = (loss_eps - loss) / epsilon
    model.W = W  # Восстанавливаем W
    
    # Градиент по b
    for j in range(b.shape[1]):
        b_eps = b.copy()
        b_eps[0, j] += epsilon
        model.b = b_eps
        probs_eps = model.softmax(model.forward(X))
        loss_eps = cross_entropy_loss(y, probs_eps)
        grad_b[0, j] = (loss_eps - loss) / epsilon
    model.b = b  # Восстанавливаем b
    
    return grad_W, grad_b



def bfgs_optimizer(model, X, y, num_iters=100):
    # Инициализируем параметры для BFGS
    n_params = model.W.size + model.b.size
    xk = np.concatenate([model.W.flatten(), model.b.flatten()])
    Hk = np.eye(n_params)
    epsilon = 1e-5
    
    for iter in range(num_iters):
        # Вычисляем градиент
        grad_W, grad_b = numerical_gradient(model, X, y, epsilon)
        grad = np.concatenate([grad_W.flatten(), grad_b.flatten()])
        
        # Проверяем условие остановки
        if np.linalg.norm(grad) < epsilon:
            print(f'Сошлось на итерации {iter}')
            break
        
        # Направление поиска
        pk = -Hk @ grad
        
        # Линейный поиск (упрощенный)
        alpha = 1e-3  # Маленький шаг
        xk_new = xk + alpha * pk
        
        # Обновление параметров
        sk = xk_new - xk
        xk = xk_new
        
        # Обновляем параметры модели
        model.W = xk[:model.W.size].reshape(model.W.shape)
        model.b = xk[model.W.size:].reshape(model.b.shape)
        
        # Вычисляем новый градиент
        grad_W_new, grad_b_new = numerical_gradient(model, X, y, epsilon)
        grad_new = np.concatenate([grad_W_new.flatten(), grad_b_new.flatten()])
        
        yk = grad_new - grad
        rho_k = 1.0 / (yk @ sk)
        I = np.eye(n_params)
        Hk = (I - rho_k * np.outer(sk, yk)) @ Hk @ (I - rho_k * np.outer(yk, sk)) + rho_k * np.outer(sk, sk)
        
        if iter % 10 == 0:
            probs = model.softmax(model.forward(X))
            loss = cross_entropy_loss(y, probs)
            print(f'Итерация {iter}, Потери: {loss}')



def lbfgs_optimizer(model, X, y, num_iters=100, m=10):
    """
    Реализует алгоритм L-BFGS для оптимизации параметров модели.
    
    Параметры:
    - model: экземпляр класса Perceptron.
    - X: входные данные (numpy массив).
    - y: метки классов (numpy массив).
    - num_iters: количество итераций.
    - m: количество последних пар (s_k, y_k) для хранения.
    """
    # Объединяем параметры модели в один вектор xk
    n_params = model.W.size + model.b.size
    xk = np.concatenate([model.W.flatten(), model.b.flatten()])
    
    # Инициализируем градиент
    epsilon = 1e-5
    grad_W, grad_b = numerical_gradient(model, X, y, epsilon)
    gk = np.concatenate([grad_W.flatten(), grad_b.flatten()])
    
    # Инициализируем списки для хранения s_k и y_k
    s_list = []
    y_list = []
    
    for iter in range(num_iters):
        # Проверяем условие сходимости
        if np.linalg.norm(gk) < epsilon:
            print(f'Сошлось на итерации {iter}')
            break
        
        # Двухцикловая рекурсия для вычисления направления поиска pk
        q = gk.copy()
        alpha_list = []
        rho_list = []
        
        # Первый цикл: идем от последнего к первому
        for i in range(len(s_list)-1, -1, -1):
            s_i = s_list[i]
            y_i = y_list[i]
            rho_i = 1.0 / (y_i @ s_i)
            rho_list.append(rho_i)
            alpha_i = rho_i * (s_i @ q)
            alpha_list.append(alpha_i)
            q = q - alpha_i * y_i
        
        # Инициализация H_0 (обычно берут скалярное произведение последних s и y)
        if len(s_list) > 0:
            gamma_k = (s_list[-1] @ y_list[-1]) / (y_list[-1] @ y_list[-1])
            Hk0 = gamma_k * np.eye(n_params)
        else:
            Hk0 = np.eye(n_params)
        
        r = Hk0 @ q
        
        # Второй цикл: идем от первого к последнему
        for i in range(len(s_list)):
            s_i = s_list[i]
            y_i = y_list[i]
            rho_i = rho_list[-(i+1)]  # Инвертируем порядок
            alpha_i = alpha_list[-(i+1)]
            beta_i = rho_i * (y_i @ r)
            r = r + s_i * (alpha_i - beta_i)
        
        pk = -r  # Направление поиска
        
        # Линейный поиск для определения шага alpha
        # Для простоты используем фиксированный шаг
        alpha = 1e-3  # Маленький фиксированный шаг
        
        # Обновляем параметры
        xk_new = xk + alpha * pk
        sk = xk_new - xk  # Изменение параметров
        
        # Обновляем параметры модели
        xk = xk_new
        model.W = xk[:model.W.size].reshape(model.W.shape)
        model.b = xk[model.W.size:].reshape(model.b.shape)
        
        # Вычисляем новый градиент
        grad_W_new, grad_b_new = numerical_gradient(model, X, y, epsilon)
        gk_new = np.concatenate([grad_W_new.flatten(), grad_b_new.flatten()])
        yk = gk_new - gk  # Разница градиентов
        
        # Обновляем списки s_k и y_k
        s_list.append(sk)
        y_list.append(yk)
        
        # Храним только последние m пар
        if len(s_list) > m:
            s_list.pop(0)
            y_list.pop(0)
        
        # Обновляем градиент для следующей итерации
        gk = gk_new
        
        # Выводим информацию каждые 10 итераций
        if iter % 10 == 0:
            probs = model.softmax(model.forward(X))
            loss = cross_entropy_loss(y, probs)
            print(f'Итерация {iter}, Потери: {loss}')
