import numpy as np

class PCA:
    def __init__(self, X):
        # Центрируем матрицу
        self.X = X - np.mean(X, axis=0)
        self.n, self.m = self.X.shape

    def pca_svd(self):
        # Основной метод PCA на основе сингулярного разложения
        # QR-алгоритм для собственных значений и векторов
        cov_matrix = np.dot(self.X.T, self.X)
        eigenvalues, eigenvectors = self.qr_algorithm(cov_matrix)
        return self.transform_data(eigenvectors)

    def fast_pca(self):
        # Метод PCA с асимптотически более быстрой реализацией
        # Используем метод мощности для вычисления наибольших собственных значений и векторов
        cov_matrix = np.dot(self.X.T, self.X)
        eigenvalues, eigenvectors = self.power_iteration(cov_matrix, num_eigenvectors=min(self.m, 10))
        return self.transform_data(eigenvectors)

    def kernel_pca(self, kernel='rbf', gamma=0.1):
        # Метод Kernel PCA для нелинейно-неразделимых данных
        K = self.calculate_kernel(self.X, kernel, gamma)
        # QR-разложение матрицы K
        eigenvalues, eigenvectors = self.qr_algorithm(K)
        return self.transform_data(eigenvectors)

    def calculate_kernel(self, X, kernel, gamma):
        if kernel == 'rbf':
            sq_dists = -2 * np.dot(X, X.T) + np.sum(X**2, axis=1).reshape(-1, 1) + np.sum(X**2, axis=1)
            K = np.exp(-gamma * sq_dists)
        elif kernel == 'polynomial':
            K = (np.dot(X, X.T) + 1) ** gamma
        elif kernel == 'sigmoid':
            K = np.tanh(gamma * np.dot(X, X.T) + 1)
        else:
            raise ValueError('Unsupported kernel type.')
        return K

    def qr_algorithm(self, A, num_iterations=500):
        # Реализация QR-алгоритма для поиска собственных значений и векторов
        Ak = np.copy(A)
        Q_total = np.eye(A.shape[0])
        for _ in range(num_iterations):
            Q, R = self.qr_decomposition(Ak)
            Ak = np.dot(R, Q)
            Q_total = np.dot(Q_total, Q)
        eigenvalues = np.diag(Ak)
        return eigenvalues, Q_total

    def qr_decomposition(self, A):
        # Реализация разложения QR методом ортогонализации Грама-Шмидта
        m, n = A.shape
        Q = np.zeros((m, n))
        R = np.zeros((n, n))
        for i in range(n):
            v = A[:, i]
            for j in range(i):
                R[j, i] = np.dot(Q[:, j], A[:, i])
                v = v - R[j, i] * Q[:, j]
            R[i, i] = np.linalg.norm(v)
            Q[:, i] = v / R[i, i]
        return Q, R

    def power_iteration(self, A, num_eigenvectors, num_iterations=500):
        # Метод мощности для вычисления наибольших собственных значений и векторов
        n, m = A.shape
        eigenvectors = np.zeros((m, num_eigenvectors))
        eigenvalues = np.zeros(num_eigenvectors)
        for i in range(num_eigenvectors):
            b = np.random.rand(m)
            for _ in range(num_iterations):
                b = np.dot(A, b)
                b = b / np.linalg.norm(b)
            eigenvalue = np.dot(b.T, np.dot(A, b))
            eigenvectors[:, i] = b
            eigenvalues[i] = eigenvalue
            A = A - eigenvalue * np.outer(b, b)  # Дефляция для следующего собственного значения
        return eigenvalues, eigenvectors

    def transform_data(self, eigenvectors):
        return np.dot(self.X, eigenvectors)

    def get_top_components(self, eigenvectors, n_components):
        return eigenvectors[:, :n_components]

    def explained_variance(self, eigenvalues):
        return eigenvalues / np.sum(eigenvalues)