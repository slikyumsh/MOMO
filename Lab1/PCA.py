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