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
        eigenvalues = np.array(eigenvalues)[idx]
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

        return eigenvalues, eigenvectors

    def find_eigenvalues_sturm(self, T, tol=1e-8):
        """
        Поиск всех собственных значений тридиагональной матрицы T с использованием метода Штурма.
        """
        n = T.shape[0]
        eigenvalues = []

        def bisect(a, b, count):
            if count == 0 or (b - a) < tol:
                if count > 0:
                    eigenvalues.extend([(a + b) / 2] * count)
                return
            mid = (a + b) / 2
            count_left = self.sturm_count(T, mid) - self.sturm_count(T, a)
            bisect(a, mid, count_left)
            bisect(mid, b, count - count_left)

        lower_bound = np.min(np.diag(T)) - np.sum(np.abs(T))
        upper_bound = np.max(np.diag(T)) + np.sum(np.abs(T))
        total_eigenvalues = self.sturm_count(T, upper_bound) - self.sturm_count(T, lower_bound)
        bisect(lower_bound, upper_bound, total_eigenvalues)
        eigenvalues.sort()
        return eigenvalues

    def sturm_count(self, T, x, tol=1e-8):
        """
        Подсчет числа собственных значений меньше x с использованием последовательности Штурма.
        """
        n = T.shape[0]
        diag = np.diag(T)
        off_diag = np.diag(T, k=-1)
        p = np.zeros(n)
        p[0] = diag[0] - x
        count = 0 if p[0] > 0 else 1

        for i in range(1, n):
            denom = p[i - 1]
            if abs(denom) < tol:
                denom = tol
            p[i] = (diag[i] - x) - (off_diag[i - 1] ** 2) / denom
            if p[i] * p[i - 1] < 0:
                count += 1
        return count

    def find_eigenvector(self, T, eigenvalue, tol=1e-8):
        """
        Поиск собственного вектора для заданного собственного значения с использованием обратных итераций.
        """
        n = T.shape[0]
        b = T - np.eye(n) * eigenvalue
        q = np.random.rand(n)
        q /= np.linalg.norm(q)

        for _ in range(1000):
            try:
                z = np.linalg.solve(b, q)
            except np.linalg.LinAlgError:
                z = np.linalg.lstsq(b, q, rcond=None)[0]
            q_new = z / np.linalg.norm(z)
            if np.linalg.norm(q - q_new) < tol:
                break
            q = q_new

        return q_new

# Ассимптотика O(nd^2 + d^3)
