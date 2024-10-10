import numpy as np

class PCA:
    def __init__(self, X):
        # Center the data
        self.X = X - np.mean(X, axis=0)
        self.n_samples, self.n_features = self.X.shape

    def pca_qr(self, n_components=None, num_iterations=1000):
        """
        PCA using the QR algorithm to compute eigenvalues and eigenvectors of the covariance matrix.
        """
        # Compute the covariance matrix
        cov_matrix = np.dot(self.X.T, self.X) / (self.n_samples - 1)

        # Use QR algorithm to compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = self.qr_algorithm(cov_matrix, num_iterations=num_iterations)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select the top n_components
        if n_components is not None:
            eigenvectors = eigenvectors[:, :n_components]
            eigenvalues = eigenvalues[:n_components]

        # Transform the data
        transformed_data = np.dot(self.X, eigenvectors)

        return transformed_data, eigenvalues, eigenvectors

    def fast_pca(self, n_components=None):
        """
        PCA using an asymptotically faster algorithm (SVD).
        """
        # Compute SVD of the centered data matrix
        U, S, VT = np.linalg.svd(self.X, full_matrices=False)
        eigenvalues = (S ** 2) / (self.n_samples - 1)
        eigenvectors = VT.T

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        U = U[:, idx]
        S = S[idx]

        # Select the top n_components
        if n_components is not None:
            eigenvectors = eigenvectors[:, :n_components]
            eigenvalues = eigenvalues[:n_components]
            U = U[:, :n_components]
            S = S[:n_components]

        # Transform the data
        transformed_data = U * S

        return transformed_data, eigenvalues, eigenvectors

    def kernel_pca(self, kernel='rbf', gamma=None, degree=3, coef0=1, n_components=None):
        """
        Kernel PCA using the specified kernel function.
        """
        # Compute the kernel matrix
        K = self._calculate_kernel(self.X, kernel=kernel, gamma=gamma, degree=degree, coef0=coef0)

        # Center the kernel matrix
        K_centered = self._center_kernel(K)

        # Compute eigenvalues and eigenvectors
        eigenvalues, eigenvectors = self._eigen_decomposition(K_centered)

        # Sort eigenvalues and eigenvectors in descending order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]

        # Select the top n_components
        if n_components is not None:
            eigenvectors = eigenvectors[:, :n_components]
            eigenvalues = eigenvalues[:n_components]

        # Normalize eigenvectors
        eigenvectors = eigenvectors / np.sqrt(eigenvalues)

        # Transform the data
        transformed_data = eigenvectors * np.sqrt(eigenvalues)

        return transformed_data, eigenvalues, eigenvectors

    def _calculate_kernel(self, X, kernel='rbf', gamma=None, degree=3, coef0=1):
        """
        Calculate the kernel matrix using the specified kernel function.
        """
        if gamma is None:
            gamma = 1.0 / self.n_features

        if kernel == 'rbf':
            sq_dists = np.sum(X**2, axis=1).reshape(-1, 1) + \
                       np.sum(X**2, axis=1) - 2 * np.dot(X, X.T)
            K = np.exp(-gamma * sq_dists)
        elif kernel == 'polynomial':
            K = (gamma * np.dot(X, X.T) + coef0) ** degree
        elif kernel == 'sigmoid':
            K = np.tanh(gamma * np.dot(X, X.T) + coef0)
        else:
            raise ValueError(f"Unsupported kernel: {kernel}")

        return K

    def _center_kernel(self, K):
        """
        Center the kernel matrix K.
        """
        n = K.shape[0]
        one_n = np.ones((n, n)) / n
        K_centered = K - one_n @ K - K @ one_n + one_n @ K @ one_n
        return K_centered

    def _eigen_decomposition(self, K):
        """
        Compute the eigenvalues and eigenvectors of a symmetric matrix K.
        """
        eigenvalues, eigenvectors = np.linalg.eigh(K)
        return eigenvalues, eigenvectors

    def qr_algorithm(self, A, num_iterations=1000):
        """
        QR algorithm for eigenvalue decomposition.
        """
        n = A.shape[0]
        Ak = A.copy()
        Q_total = np.eye(n)
        for _ in range(num_iterations):
            Q, R = self.qr_decomposition(Ak)
            Ak = R @ Q
            Q_total = Q_total @ Q
        eigenvalues = np.diag(Ak)
        eigenvectors = Q_total
        return eigenvalues, eigenvectors

    def qr_decomposition(self, A):
        """
        QR decomposition using Householder reflections.
        """
        n = A.shape[0]
        Q = np.eye(n)
        R = A.copy()
        for i in range(n - 1):
            x = R[i:, i]
            e1 = np.zeros_like(x)
            e1[0] = np.linalg.norm(x)
            u = x - e1
            v = u / np.linalg.norm(u)
            H = np.eye(n)
            H_i = np.eye(len(x)) - 2.0 * np.outer(v, v)
            H[i:, i:] = H_i
            R = H @ R
            Q = Q @ H.T
        return Q, R