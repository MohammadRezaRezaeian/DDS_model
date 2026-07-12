import numpy as np

class DiagnosticsCalculator:
    def __init__(self, model):
        self.model = model

    def get_spectral_radius(self) -> float:
        r"""
        Metric 1: Static System Stability (Last Matrix Spectral Radius)
        
        WHAT IT IS:
        Extracts only the final lag matrix (tau = -1) and calculates its maximum eigenvalue,
        bypassing the need to average across all lags.
        """
        # Extract the most recent lag matrix
        last_matrix = self.model.mu_tensor[:, :, -1]
        
        eigenvalues = np.linalg.eigvals(last_matrix)
        total_spectral_radius = np.max(np.abs(eigenvalues))
            
        return float(np.clip(total_spectral_radius, 0.0, 5.0))

    def calculate_bic(self, n_observations: int, mse_loss: float) -> float:
        """
        Metric 2: Bayesian Information Criterion (BIC)
        """
        k_params = self.model.N * self.model.N * self.model.L * 2 
        
        if mse_loss <= 0: return 0.0
        
        return (k_params * np.log(n_observations)) + (n_observations * np.log(mse_loss))
    
    def get_koopman_spectral_radius(self, max_iter=100, tol=1e-6) -> float:
        """
        Metric 3: Dominant Dynamic via Implicit Power Iteration
        """
        N = self.model.N
        L = self.model.L

        x = np.random.randn(N * L)
        x /= np.linalg.norm(x)

        last = 0.0
        y = np.zeros_like(x)
        
        for _ in range(max_iter):
            top_block = np.zeros(N)
            for tau in range(L):
                top_block += self.model.mu_tensor[:, :, tau] @ x[tau*N : (tau+1)*N]
                
            y[:N] = top_block
            if L > 1:
                y[N:] = x[:-N]
                
            norm_y = np.linalg.norm(y)
            if abs(norm_y - last) < tol:
                break
                
            x = y / norm_y
            last = norm_y

        return float(np.clip(norm_y, 0, 5))
    
    def get_stochastic_spectral_radius(self) -> float:
        """
        Metric 4: Uncertainty & Fragility (Last Matrix Stochastic Spectral Radius)
        
        WHAT IT IS:
        Finds the exact stochastic eigenvalue variance for the final lag matrix only.
        """
        # Extract only the final lag's mean and variance matrices
        M_last = self.model.mu_tensor[:, :, -1]
        S2_last = self.model.sigma_sq_tensor[:, :, -1]
        
        # Calculate exact eigenvalues and right eigenvectors for this matrix
        eigenvalues, V_right = np.linalg.eig(M_last)
        
        try:
            # Calculate left eigenvectors
            U_left = np.linalg.inv(V_right)
        except np.linalg.LinAlgError:
            U_left = np.zeros_like(V_right)

        # Find the dominant eigenvalue (highest absolute value)
        k = np.argmax(np.abs(eigenvalues))
        expected_lambda = eigenvalues[k]
        
        # Extract corresponding left and right eigenvectors for the dominant mode
        u_k = U_left[k, :]
        v_k = V_right[:, k]
        
        # Calculate sensitivity and variance via Perturbation Theory
        sensitivity_matrix = np.outer(u_k, v_k)
        eigenvalue_variance = np.sum((np.abs(sensitivity_matrix) ** 2) * S2_last)
        
        # Sample a possible reality based on the expected dynamic and its variance
        sampled_radius = np.random.normal(
            loc=np.abs(expected_lambda), 
            scale=np.sqrt(eigenvalue_variance)
        )
        
        return float(np.clip(np.abs(sampled_radius), 0.0, 5.0))