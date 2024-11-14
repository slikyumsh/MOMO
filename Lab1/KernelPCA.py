import numpy as np

def center_kernel(K):
    n = K.shape[0]
    one_n = np.ones((n, n)) / n
    K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
    return K_centered

def power_iteration(A, num_iterations=1000, tol=1e-6):
    """
    Степенной метод для нахождения наибольшего собственного значения и собственного вектора.

    Параметры:
    - A: симметричная матрица
    - num_iterations: максимальное число итераций
    - tol: допустимая погрешность

    Возвращает:
    - наибольшее собственное значение
    - соответствующий нормированный собственный вектор
    """
    n = A.shape[0]
    b_k = np.random.rand(n)
    b_k = b_k / np.linalg.norm(b_k)
    for _ in range(num_iterations):
        b_k1 = A @ b_k
        b_k1_norm = np.linalg.norm(b_k1)
        b_k1 = b_k1 / b_k1_norm
        # Проверка сходимости
        if np.allclose(b_k, b_k1, atol=tol):
            break
        b_k = b_k1
    eigenvalue = b_k.T @ A @ b_k
    return eigenvalue, b_k

def compute_top_eigenpairs(A, n_components, num_iterations=1000, tol=1e-6):
    """
    Нахождение нескольких наибольших собственных значений и векторов.

    Параметры:
    - A: симметричная матрица
    - n_components: количество требуемых собственных пар
    - num_iterations: максимальное число итераций для степенного метода
    - tol: допустимая погрешность

    Возвращает:
    - массив собственных значений
    - матрицу собственных векторов
    """
    n = A.shape[0]
    eigenvalues = []
    eigenvectors = []
    A_copy = A.copy()
    for _ in range(n_components):
        eigenvalue, eigenvector = power_iteration(A_copy, num_iterations, tol)
        eigenvalues.append(eigenvalue)
        eigenvectors.append(eigenvector)
        A_copy -= eigenvalue * np.outer(eigenvector, eigenvector)
    eigenvalues = np.array(eigenvalues)
    eigenvectors = np.array(eigenvectors).T  # Транспонируем для согласованности
    return eigenvalues, eigenvectors

def kernel_pca(X, kernel_function, n_components=2, num_iterations=1000, tol=1e-6):
    """
    Реализация Kernel PCA с использованием собственного метода нахождения собственных чисел.
    """

    K = kernel_function(X)
    K_centered = center_kernel(K)
    eigvals, eigvecs = compute_top_eigenpairs(K_centered, n_components, num_iterations, tol)
    for i in range(n_components):
        eigvecs[:, i] /= np.sqrt(eigvals[i])

    X_pc = K_centered @ eigvecs

    return X_pc



def polynomial_kernel(X, degree=3, coef0=1):
    """
    Полиномиальное ядро.

    K(x, y) = (x^T y + coef0)^degree
    """
    return (X @ X.T + coef0) ** degree



def rbf_kernel(X, gamma=None):
    """
    RBF ядро.

    K(x, y) = exp(-gamma * ||x - y||^2)
    """
    if gamma is None:
        gamma = 1 / X.shape[1]
    sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + \
               np.sum(X**2, axis=1) - 2 * X @ X.T
    K = np.exp(-gamma * sq_dists)
    return K



def sigmoid_kernel(X, kappa=1, theta=0):
    """
    Сигмоидное ядро.

    K(x, y) = tanh(kappa * x^T y + theta)
    """
    return np.tanh(kappa * X @ X.T + theta)



def laplacian_kernel(X, gamma=None):
    """
    Лапласовское ядро.

    K(x, y) = exp(-gamma * ||x - y||_1)
    """
    if gamma is None:
        gamma = 1 / X.shape[1]
    manhattan_dists = np.sum(np.abs(X[:, np.newaxis] - X[np.newaxis, :]), axis=2)
    K = np.exp(-gamma * manhattan_dists)
    return K



