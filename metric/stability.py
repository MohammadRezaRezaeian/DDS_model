import numpy as np

class DiagnosticsCalculator:
    def __init__(self, model):
        self.model = model

    def get_spectral_radius(self) -> float:
        r"""
        Metric 1: Deterministic System Stability (Companion Matrix Spectral Radius)
        
        WHAT IT IS:
        Builds a full Companion Matrix out of all $L$ lag tensors and calculates 
        every exact eigenvalue ($\lambda$) for the entire system. It returns the 
        Spectral Radius ($\rho$), which is the absolute maximum eigenvalue.
        
        MEANING:
        Dictates how the system behaves as time moves forward to infinity.
        * $\rho < 1.0$: Stable / Mean-Reverting. Predictions settle to a baseline.
        * $\rho == 1.0$: Random Walk. The system preserves momentum forever.
        * $\rho > 1.0$: Explosive. Tiny inputs compound infinitely (feedback loop).
        
        HOW TO USE IT:
        Monitor this during training loops. If it consistently drifts above 1.0, 
        your optimizer is learning dangerous feedback loops. If it stays perfectly 
        near 1.0, the model considers the market purely random.
        """
        N = self.model.N
        L = self.model.L
        
        # If there's only 1 lag, the companion matrix is just the lag 1 matrix
        if L == 1:
            eigenvalues = np.linalg.eigvals(self.model.mu_tensor[:, :, 0])
            return float(np.clip(np.max(np.abs(eigenvalues)), 0.0, 5.0))
            
        # 1. Initialize the full Companion Matrix (Size: N*L x N*L)
        companion_matrix = np.zeros((N * L, N * L))
        
        # 2. Fill the top row blocks with the lag weights A_1, A_2, ..., A_L
        for tau in range(L):
            # mu_tensor[:, :, tau] represents the weight matrix for lag = tau+1
            start_col = tau * N
            end_col = (tau + 1) * N
            companion_matrix[0:N, start_col:end_col] = self.model.mu_tensor[:, :, tau]
            
        # 3. Fill the sub-diagonal with Identity matrices to handle the time shift
        identity_block = np.eye(N * (L - 1))
        companion_matrix[N:, 0:N*(L-1)] = identity_block
        
        # 4. Calculate the true systemic eigenvalues
        eigenvalues = np.linalg.eigvals(companion_matrix)
        total_spectral_radius = np.max(np.abs(eigenvalues))
            
        return float(np.clip(total_spectral_radius, 0.0, 5.0))

    def calculate_bic(self, n_observations: int, mse_loss: float) -> float:
        """
        Metric 2: Bayesian Information Criterion (BIC)
        
        WHAT IT IS:
        A statistical metric for model selection. It calculates the goodness-of-fit 
        (MSE) but heavily penalizes the model for having too many parameters.
        
        MEANING:
        More parameters (higher lag depth or more assets) will almost always reduce 
        Train MSE because the model memorizes noise. BIC guards against this overfitting.
        
        HOW TO USE IT:
        Use it to compare different configurations. If you increase lag depth from 
        2 to 10, MSE will drop. But if BIC increases, the slight reduction in error 
        was not worth the massive increase in complexity. Always choose the model 
        with the lowest BIC.
        """
        # Calculate total parameters: N targets * N predictors * L lags * 2 (mu and sigma)
        k_params = self.model.N * self.model.N * self.model.L * 2 
        
        if mse_loss <= 0: return 0.0
        
        return (k_params * np.log(n_observations)) + (n_observations * np.log(mse_loss))
    
    def get_koopman_spectral_radius(self, max_iter=100, tol=1e-6) -> float:
        """
        Metric 3: Dominant Dynamic via Power Iteration
        
        WHAT IT IS:
        Builds the exact same Companion Matrix as Metric 1, but uses the Power 
        Iteration method to approximate only the single largest eigenvalue instead 
        of solving for all of them exactly.
        
        MEANING:
        Mathematically represents the same dominant dynamic (the mode that grows 
        the fastest or decays the slowest) as Metric 1.
        
        HOW TO USE IT:
        Use this for performance optimization. Calculating exact eigenvalues for 
        massive matrices (e.g., 500 assets, 100 lags) will crash your memory. 
        Power iteration is computationally cheap and runs almost instantly for 
        large-scale models.
        """
        N = self.model.N
        L = self.model.L
        K = np.zeros((N * L, N * L))

        # Build Companion Matrix
        for tau in range(L):
            K[:N, tau*N:(tau+1)*N] = self.model.mu_tensor[:, :, tau]
        for i in range(1, L):
            K[i*N:(i+1)*N, (i-1)*N:i*N] = np.eye(N)

        # Initialize random vector
        x = np.random.randn(N * L)
        x /= np.linalg.norm(x)

        # Power Iteration Loop
        last = 0.0
        for _ in range(max_iter):
            y = K @ x
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
        Collapses lag tensors into an expected system matrix, calculates eigenvalues 
        and left/right eigenvectors, and uses First-Order Perturbation Theory to 
        determine how sensitive the dominant eigenvalue is to the model's variance.
        
        MEANING:
        Evaluates risk. Even if the expected model is perfectly stable, your Bayesian 
        model contains uncertainty (sigma_sq_tensor). This metric asks: "Given our 
        margin of error, could the system accidentally explode?"
        
        HOW TO USE IT:
        Compare this to Metric 1 (Deterministic Radius). 
        * If Metric 1 = 0.90 and Metric 4 = 0.92: Model is robust and confident.
        * If Metric 1 = 0.90 and Metric 4 = 1.50: Model is fragile. You have massive 
          uncertainty on a highly sensitive causal weight. Increase regularization.
        """
        # Collapse tensors to expected 2D matrices across all lags
        M_expected = np.sum(self.model.mu_tensor, axis=2)
        S2_expected = np.sum(self.model.sigma_sq_tensor, axis=2)
        
        # Calculate exact eigenvalues and right eigenvectors
        eigenvalues, V_right = np.linalg.eig(M_expected)
        
        try:
            # Calculate left eigenvectors
            U_left = np.linalg.inv(V_right)
        except np.linalg.LinAlgError:
            return 0.0

        # Find the dominant eigenvalue (highest absolute value)
        k = np.argmax(np.abs(eigenvalues))
        expected_lambda = eigenvalues[k]
        
        # Extract corresponding left and right eigenvectors
        u_k = U_left[k, :]
        v_k = V_right[:, k]
        
        # Calculate sensitivity and variance via Perturbation Theory
        sensitivity_matrix = np.outer(u_k, v_k)
        eigenvalue_variance = np.sum((np.abs(sensitivity_matrix) ** 2) * S2_expected)
        
        # Sample a possible reality based on the expected dynamic and its variance
        sampled_radius = np.random.normal(
            loc=np.abs(expected_lambda), 
            scale=np.sqrt(eigenvalue_variance)
        )
        
        return float(np.clip(np.abs(sampled_radius), 0.0, 3.0))