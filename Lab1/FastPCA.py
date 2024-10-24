import numpy as np

class PCASturm:
    def __init__(self, X):
        # Центрируем данные и нормализуем их
        self.X = (X - np.mean(X, axis=0)) / np.std(X, axis=0)
        self.n_samples, self.n_features = self.X.shape

    def pca_sturm(self, n_components=None, tol=1e-8):
        """
        PCA с использованием метода Штурма для нахождения собственных значений и векторов ковариационной матрицы.
        """
        # Вычисляем ковариационную матрицу
        cov_matrix = np.dot(self.X.T, self.X) / (self.n_samples - 1)

        # Преобразуем ковариационную матрицу в тридиагональную форму
        tridiagonal_matrix, Q_tridiagonal = self.tridiagonalize(cov_matrix)

        # Используем метод Штурма для нахождения собственных значений и векторов
        eigenvalues, eigenvectors = self.sturm_method(tridiagonal_matrix, tol=tol)

        # Сортируем собственные значения и векторы по убыванию
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = Q_tridiagonal @ eigenvectors[:, idx]

        # Выбираем top n_components
        if n_components is not None:
            eigenvectors = eigenvectors[:, :n_components]
            eigenvalues = eigenvalues[:n_components]

        # Преобразуем данные
        transformed_data = np.dot(self.X, eigenvectors)

        return transformed_data, eigenvalues, eigenvectors

    def tridiagonalize(self, A):
        """
        Преобразование симметричной матрицы A в тридиагональную форму с использованием отражений Хаусхолдера.
        """
        n = A.shape[0]
        Q = np.eye(n)
        T = A.copy()

        for k in range(n - 2):
            x = T[k+1:, k]
            e1 = np.zeros_like(x)
            e1[0] = np.linalg.norm(x)
            u = x - e1
            v = u / np.linalg.norm(u)
            H_k = np.eye(n)
            H_k[k+1:, k+1:] -= 2.0 * np.outer(v, v)
            T = H_k @ T @ H_k
            Q = Q @ H_k

        return T, Q

    def sturm_method(self, T, tol=1e-8):
        """
        Метод Штурма для нахождения собственных значений и векторов тридиагональной матрицы T.
        """
        n = T.shape[0]
        eigenvalues = self.find_eigenvalues_sturm(T, tol)
        eigenvectors = np.zeros((n, n))

        for i, eigenvalue in enumerate(eigenvalues):
            eigenvectors[:, i] = self.find_eigenvector(T, eigenvalue, tol)

        return np.array(eigenvalues), eigenvectors

    def find_eigenvalues_sturm(self, T, tol=1e-8):
        """
        Поиск всех собственных значений тридиагональной матрицы T с использованием метода Штурма.
        """
        n = T.shape[0]
        eigenvalues = []
        # Простая реализация деления отрезка до нужной точности для поиска всех собственных значений
        min_value = np.min(np.diag(T)) - 1
        max_value = np.max(np.diag(T)) + 1
        while len(eigenvalues) < n:
            mid_value = (min_value + max_value) / 2
            if self.sturm_count(T, mid_value) < n:
                min_value = mid_value
            else:
                max_value = mid_value
            if abs(max_value - min_value) < tol:
                eigenvalues.append(mid_value)
                min_value = mid_value + tol
                max_value = np.max(np.diag(T)) + 1
        return sorted(eigenvalues)

    def sturm_count(self, T, x):
        """
        Подсчёт числа изменений знака для метода Штурма.
        """
        n = T.shape[0]
        p0 = 1
        p1 = T[0, 0] - x
        count = 0 if p1 >= 0 else 1

        for i in range(1, n):
            p2 = (T[i, i] - x) - (T[i, i - 1] ** 2) / p1
            if p2 < 0 and p1 >= 0 or p2 >= 0 and p1 < 0:
                count += 1
            p1 = p2

        return count

    def find_eigenvector(self, T, eigenvalue, tol=1e-8):
        """
        Поиск собственных векторов для заданного собственного значения.
        """
        n = T.shape[0]
        b = T - np.eye(n) * eigenvalue
        q = np.random.rand(n)
        q /= np.linalg.norm(q)

        for _ in range(1000):
            q_new = np.linalg.solve(b, q)
            q_new /= np.linalg.norm(q_new)
            if np.linalg.norm(q - q_new) < tol:
                break
            q = q_new

        return q


# Ассимптотика O(nd^2 + d^3)