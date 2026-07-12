import numpy as np

class DiagnosticsCalculator:
    def __init__(self, model):
        self.model = model

    def get_spectral_radius(self) -> float:
        r"""
        Metric 1: Static System Stability (Averaged Lag Matrix Spectral Radius)
        
        WHAT IT IS:
        Calculates the eigenvalues for every individual lag matrix, aligns them 
        by magnitude, averages them across all L lags, and finds the maximum.
        """
        L = self.model.L
        N = self.model.N
        
        all_eigenvalues = np.zeros((L, N), dtype=complex)
        
        for tau in range(L):
            eigvals = np.linalg.eigvals(self.model.mu_tensor[:, :, tau])
            # Sort eigenvalues by magnitude to meaningfully align and average them
            all_eigenvalues[tau, :] = eigvals[np.argsort(np.abs(eigvals))]
        
        eighens = eigvals = np.linalg.eigvals(self.model.mu_tensor[:, :, -1])
        # Average the eigenvalues across all lags
        avg_eigenvalues = np.mean(eighens, axis=0)
        
        # Find the maximum of the averaged eigenvalues
        total_spectral_radius = np.max(np.abs(avg_eigenvalues))
            
        return float(np.clip(total_spectral_radius, 0.0, 5.0))

    def calculate_bic(self, n_observations: int, mse_loss: float) -> float:
        """
        Metric 2: Bayesian Information Criterion (BIC)
        """
        # Calculate total parameters: N targets * N predictors * L lags * 2 (mu and sigma)
        k_params = self.model.N * self.model.N * self.model.L * 2 
        
        if mse_loss <= 0: return 0.0
        
        return (k_params * np.log(n_observations)) + (n_observations * np.log(mse_loss))
    
    def get_koopman_spectral_radius(self, max_iter=100, tol=1e-6) -> float:
        """
        Metric 3: Dominant Dynamic via Implicit Power Iteration
        """
        N = self.model.N
        L = self.model.L

        # Initialize random vector
        x = np.random.randn(N * L)
        x /= np.linalg.norm(x)

        # Power Iteration Loop
        last = 0.0
        y = np.zeros_like(x)
        
        for _ in range(max_iter):
            # 1. OPTIMIZATION: Calculate the top block implicitly
            top_block = np.zeros(N)
            for tau in range(L):
                # Multiply each lag's weights by the corresponding slice of the vector
                top_block += self.model.mu_tensor[:, :, tau] @ x[tau*N : (tau+1)*N]
                
            # 2. Shift the remaining blocks down (simulates the sub-diagonal Identity matrix)
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
        Metric 4: Uncertainty & Fragility (Analytical Stochastic Spectral Radius)
        
        WHAT IT IS:
        Calculates eigenvalues and perturbation variance for every individual lag matrix, 
        averages both the eigenvalues and variances across all lags, and samples 
        from the maximum averaged eigenmode.
        """
        L = self.model.L
        N = self.model.N
        
        all_eigenvalues = np.zeros((L, N), dtype=complex)
        all_variances = np.zeros((L, N))
        
        for tau in range(L):
            M = self.model.mu_tensor[:, :, tau]
            S2 = self.model.sigma_sq_tensor[:, :, tau]
            
            # Calculate exact eigenvalues and right eigenvectors for this lag
            eigenvalues, V_right = np.linalg.eig(M)
            
            try:
                # Calculate left eigenvectors
                U_left = np.linalg.inv(V_right)
            except np.linalg.LinAlgError:
                U_left = np.zeros_like(V_right)
                
            # Sort to align eigenvalues by magnitude across lags
            sort_idx = np.argsort(np.abs(eigenvalues))
            all_eigenvalues[tau, :] = eigenvalues[sort_idx]
            
            # Calculate perturbation variance for each aligned eigenmode
            for i, idx in enumerate(sort_idx):
                u_k = U_left[idx, :]
                v_k = V_right[:, idx]
                sensitivity_matrix = np.outer(u_k, v_k)
                mode_variance = np.sum((np.abs(sensitivity_matrix) ** 2) * S2)
                all_variances[tau, i] = mode_variance

        # Average the eigenvalues and their variances across all lags
        avg_eigenvalues = np.mean(all_eigenvalues, axis=0)
        avg_variances = np.mean(all_variances, axis=0)

        # Find the maximum of the averaged eigenvalues
        k = np.argmax(np.abs(avg_eigenvalues))
        expected_lambda = avg_eigenvalues[k]
        eigenvalue_variance = avg_variances[k]
        
        # Sample a possible reality based on the expected dynamic and its variance
        sampled_radius = np.random.normal(
            loc=np.abs(expected_lambda), 
            scale=np.sqrt(eigenvalue_variance)
        )
        
        return float(np.clip(np.abs(sampled_radius), 0.0, 3.0))