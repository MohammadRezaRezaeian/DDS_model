import numpy as np
import pickle
import os
import time

from component_factory import ComponentFactory
from metric.stability import DiagnosticsCalculator



class ModelEngine:
    def __init__(self, l_params: dict, m_params: dict):
        self.l_params = l_params
        self.m_params = m_params
        
        self.metric = ComponentFactory.create_metric(m_params["metric_name"])
        self.optimizer = ComponentFactory.create_optimizer(m_params["optimizer_name"])
        self.model = ComponentFactory.create_model(
            m_params["model_name"], self.metric, self.optimizer, m_params["params"]
        )
        self.diagnostics = DiagnosticsCalculator(self.model)

        self.L = self.model.L
        self.train_metrics = [] 
        self.history_buffer = None
        
        self._set_iteration_method()

    def _set_iteration_method(self):
        method = self.l_params.get("itteration_method", "sequential")
        if hasattr(self.model, 'set_iteration_method'):
            self.model.set_iteration_method(method)
        else:
            self.model.iteration_method = method

    def execute_pipeline(self, data_matrix: np.ndarray) -> dict:
        T_total = data_matrix.shape[0]
        
        warmUp_idx = self.l_params["warmUp"]
        split_idx = int(T_total * self.l_params["split_ratio"])
        future_days = self.l_params["future_days"]
        
        train_val_ratio = 0.80
        train_end_idx = warmUp_idx + int((split_idx - warmUp_idx) * train_val_ratio)
        
        warmup_data = data_matrix[:warmUp_idx]
        train_subset = data_matrix[warmUp_idx:train_end_idx]
        test_data = data_matrix[split_idx:]
        test_train = data_matrix[split_idx - self.L:] 
        
        print("[*] PHASE 0: Running OLS Warmup...")
        self.warmup(warmup_data)

        N_epochs = self.l_params.get("N_epochs", 1)
        print(f"[*] PHASE 1: Training Model for {N_epochs} Epochs...")
        
        post_warmup_buffer = np.copy(self.history_buffer)
        
        # --- Delegated Phase 1 ---
        final_train_preds = self._train_epochs(
            N_epochs=N_epochs,
            post_warmup_buffer=post_warmup_buffer,
            train_subset=train_subset,
            data_matrix=data_matrix,
            warmUp_idx=warmUp_idx,
            split_idx=split_idx
        )

        print("[*] PHASE 2: Predicting Test Data (Anchored & Autoregressive)...")
        test_preds = self.predict(test_data, autoregressive=False)
        test_preds_autoregressive = self.predict(test_data, autoregressive=True)

        print("[*] PHASE 3: Catch-up Updating on Test Data...")
        self.update(test_train, is_training=True)

        print(f"[*] PHASE 4: Forecasting {future_days} days into the future...")
        future_preds = self.predict(np.zeros((future_days, data_matrix.shape[1])), autoregressive=True)

        return {
            "train_preds": final_train_preds,
            "test_preds": test_preds,
            "test_preds_autoregressive": test_preds_autoregressive,
            "future_preds": future_preds,
            "warmUp_idx": warmUp_idx,
            "split_idx": split_idx,
            "metrics_history": self.train_metrics 
        }

    def _chunk_data(self, data: np.ndarray, num_chunks: int, initial_buffer: np.ndarray):
        """
        Generator that yields chunks of the dataset based on the specified iteration method, 
        paired with their true historical buffer to prevent price-teleportation shocks.
        Each chunk is strictly divided chronologically into train and validation subsets.
        """
        method = self.l_params.get("itteration_method", "sequential")
        base_chunk_size = self.l_params.get("chunk_size", 100)
        
        # New parameter: what percentage of the chunk should be held out for validation?
        val_ratio = self.l_params.get("val_ratio", 0.2) 
        
        if method == "random":
            # ---------------------------------------------------------
            # 1. RANDOM: Stochastic cuts anywhere in the acceptable timeline
            # ---------------------------------------------------------
            for _ in range(num_chunks):
                if len(data) <= self.L + base_chunk_size:
                    start_idx = self.L 
                else:
                    start_idx = np.random.randint(self.L, len(data) - base_chunk_size + 1)
                
                true_history = data[start_idx - self.L : start_idx]
                full_chunk = data[start_idx : start_idx + base_chunk_size]
                
                # Split chronologically
                split_idx = int(len(full_chunk) * (1.0 - val_ratio))
                epoch_train = full_chunk[:split_idx]
                epoch_val = full_chunk[split_idx:]
                
                yield epoch_train, true_history, epoch_val

        elif method == "split":
            # ---------------------------------------------------------
            # 2. SPLIT: Contiguous, non-overlapping sequential blocks
            # ---------------------------------------------------------
            chunk_size = len(data) // num_chunks
            current_history = np.copy(initial_buffer)
            
            for i in range(num_chunks):
                start_idx = i * chunk_size
                end_idx = (i + 1) * chunk_size if i < (num_chunks - 1) else len(data)
                
                full_chunk = data[start_idx:end_idx]
                
                # Split chronologically
                split_idx = int(len(full_chunk) * (1.0 - val_ratio))
                epoch_train = full_chunk[:split_idx]
                epoch_val = full_chunk[split_idx:]
                
                yield epoch_train, current_history, epoch_val
                
                # IMPORTANT: History tracking for the NEXT chunk must use the FULL current chunk
                # so the model doesn't experience a temporal gap across the validation set.
                if len(full_chunk) >= self.L:
                    current_history = full_chunk[-self.L:]
                else:
                    current_history = np.vstack([current_history, full_chunk])[-self.L:]

        elif method == "sequential":
            # ---------------------------------------------------------
            # 3. SEQUENTIAL: Sliding window moving evenly from start to end
            # ---------------------------------------------------------
            if len(data) <= self.L + base_chunk_size:
                step_size = 0.0
            else:
                available_timeline = len(data) - self.L - base_chunk_size
                step_size = available_timeline / max(1, (num_chunks - 1))
            
            for i in range(num_chunks):
                start_idx = self.L + int(i * step_size)
                
                true_history = data[start_idx - self.L : start_idx]
                full_chunk = data[start_idx : start_idx + base_chunk_size]
                
                # Split chronologically
                split_idx = int(len(full_chunk) * (1.0 - val_ratio))
                epoch_train = full_chunk[:split_idx]
                epoch_val = full_chunk[split_idx:]
                
                yield epoch_train, true_history, epoch_val
                
        else:
            raise ValueError(f"Unknown itteration_method: '{method}'. Valid options are 'random', 'split', or 'sequential'.")


    def _train_epochs(self, N_epochs: int, post_warmup_buffer: np.ndarray, 
                      train_subset: np.ndarray, data_matrix: np.ndarray,
                      warmUp_idx: int, split_idx: int) -> list:
        """
        Private method to handle continuous time-series learning.
        Consumes data from the chunk generator alongside its perfectly aligned historical buffer.
        """
        self.train_metrics = []

        # Loop directly over the yielded data stream AND its matching history buffer
        for epoch, (epoch_subset, hist_buffer, epoch_val_subset) in enumerate(self._chunk_data(train_subset, N_epochs, post_warmup_buffer), start=1):
            start_time = time.time()
            
            # Apply the correct history buffer for this specific timeline segment
            self.history_buffer = np.copy(hist_buffer)
            
            # 1. Forward pass: Train strictly on this chunk
            _, train_step_metrics = self.update(epoch_subset, is_training=True)
            
            # 2. Forward pass: Validation
            pre_val_buffer = np.copy(self.history_buffer)
            self.history_buffer = np.copy(train_subset[-self.L:])
            _, validation_step_metrics = self.update(epoch_val_subset, is_training=False)
            self.history_buffer = pre_val_buffer
            
            # ---------------------------------------------------------
            # Extract aggregated metrics natively from the model's report
            # ---------------------------------------------------------
            if train_step_metrics:
                t_loss = float(np.mean([m['MSE_Loss'] for m in train_step_metrics]))
                t_var = float(np.mean([m['MSE_var'] for m in train_step_metrics]))
                t_k_trace = float(np.mean([m['Kalman_Trace'] for m in train_step_metrics]))
                t_spectral = train_step_metrics[-1]['Spectral_Radius']
                t_stochastiic_spectral = train_step_metrics[-1]['Stochastic_Spectral_Radius']
                t_koopman = train_step_metrics[-1]['Koopman_Radius']
                t_bic = train_step_metrics[-1]['BIC']
            else:
                t_loss, t_var, t_k_trace, t_spectral, t_stochastiic_spectral, t_bic = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
                
            

            t_metrics = {
                "Loss": t_loss,
                "Var": t_var,
                "Kalman_Trace": t_k_trace,
                "Spectral_Radius": t_spectral,
                "Stochastic_Spectral_Radius": t_stochastiic_spectral,
                "Koopman_Radius": t_koopman,
                "BIC": t_bic,
                "Koopman_Spectral_Radius": t_koopman
            }

            # ---------------------------------------------------------
            # Extract aggregated metrics natively from the model's report i Validation
            # ---------------------------------------------------------
            if validation_step_metrics:
                v_loss = float(np.mean([m['MSE_Loss'] for m in validation_step_metrics]))
                v_var = float(np.mean([m['MSE_var'] for m in validation_step_metrics]))
                # v_k_trace = float(np.mean([m['Kalman_Trace'] for m in train_step_metrics]))
                # v_spectral = train_step_metrics[-1]['Spectral_Radius']
                # v_bic = train_step_metrics[-1]['BIC']
            else:
                v_loss, v_var, v_k_trace, v_spectral, v_bic = 0.0, 0.0, 0.0, 0.0, 0.0
                
            # v_koopman = self.diagnostics.get_koopman_spectral_radius() if self.diagnostics else 0.0

            # v_metrics = {
            #     "Loss": v_loss,
            #     "Kalman_Trace": v_k_trace,
            #     "Spectral_Radius": v_spectral,
            #     "BIC": v_bic,
            #     "Koopman_Radius": v_koopman
            # }
            
            # v_loss = self._get_val_loss(val_preds, val_subset)
            self.train_metrics.append(t_metrics)
            
            loss_name = self.m_params.get("metric_name", "Loss")
            
            print(f"    Chunk Sequence {epoch}/{N_epochs} | "
                  f"Train {loss_name}: {t_loss:.6f}, Var: {t_var:.6f} | "
                  f"Val {loss_name}: {v_loss:.6f}, Var: {v_var:.6f} | "
                  f"Kalman Trace: {t_k_trace:.4f} | "
                  f"S Radius: {t_spectral:.4f} | "
                  f"K Radius: {t_koopman:.4f} | "
                  f"Mu Avg: {np.mean(self.model.mu_tensor):.6f} | "
                  f"S Avg: {np.mean(self.model.sigma_sq_tensor):.6f} | "
                  f"{time.time() - start_time:.2f} s.")

        # ---------------------------------------------------------
        # Final clean prediction sweep for the reporter
        # ---------------------------------------------------------
        print("    [*] Gathering unified Phase 1 predictions for reporter...")
        full_train_block = data_matrix[warmUp_idx:split_idx]
        
        temp_buffer = np.copy(self.history_buffer)
        self.history_buffer = np.copy(post_warmup_buffer)
        final_train_preds = self.predict(full_train_block, autoregressive=False)
        self.history_buffer = temp_buffer 

        return final_train_preds

    def warmup(self, warmup_data: np.ndarray):
        self.model.warmup(warmup_data)
        self.history_buffer = np.copy(warmup_data[-self.model.L:])

    def update(self, data: np.ndarray, is_training: bool = True) -> tuple:
        """Updates the model by passing the full chunk down to the original model."""
        full_data = np.vstack([self.history_buffer, data])
        
        # ---> FIX: Turn diagnostics ON so bayesian_tensor generates the report natively
        preds, step_metrics = self.model.update(
            data=full_data, 
            diagnostics=self.diagnostics, 
            is_training=is_training 
        )
        
        self.history_buffer = np.copy(full_data[-self.model.L:])
        return preds, step_metrics

    def predict(self, data: np.ndarray, autoregressive: bool = False) -> list:
        preds = []
        auto_buffer = np.copy(self.history_buffer)
        full_data = np.vstack([self.history_buffer, data])
        
        for t in range(self.model.L, len(full_data)):
            if autoregressive:
                pred = self.model.predict(auto_buffer)
                
                # Defensively strip batch dimensions
                pred_arr = np.array(pred)
                while pred_arr.ndim > 2:
                    pred_arr = pred_arr[0]
                if pred_arr.ndim == 2:
                    pred_arr = pred_arr[0]
                    
                auto_buffer = np.vstack([auto_buffer[1:], pred_arr])
            else:
                # ---> THE FIX: Slice the specific L-depth window <---
                seq = full_data[t - self.model.L : t]
                pred = self.model.predict(seq)
                
            preds.append(pred)

        return preds

    def _calculate_metrics(self, preds: list, target_data: np.ndarray, is_training: bool) -> dict:
        if len(preds) == 0 or len(target_data) == 0:
            return {"Loss": 0.0}
        
        min_len = min(len(preds), len(target_data))
        preds_arr = np.array(preds[-min_len:])
        target_arr = target_data[-min_len:]
        
        if preds_arr.ndim == 3:
            preds_arr = preds_arr[:, 0, :]
        
        if hasattr(self.metric, 'metric'):
            loss_val = float(self.metric.metric(target_arr, preds_arr))
        else:
            loss_val = float(np.mean((preds_arr - target_arr) ** 2))

        metrics_dict = {"Loss": loss_val}
        
        if is_training and self.diagnostics:
            n_obs = len(target_arr)
            metrics_dict["BIC"] = self.diagnostics.calculate_bic(n_observations=n_obs, mse_loss=loss_val)
            metrics_dict["Koopman_Radius"] = self.diagnostics.get_koopman_spectral_radius()
            metrics_dict["Stochastic_Radius"] = self.diagnostics.get_stochastic_spectral_radius()
            metrics_dict["Spectral_Radius"] = self.diagnostics.get_spectral_radius()

        return metrics_dict
    
    def _get_val_loss(self, preds: list, target_data: np.ndarray) -> float:
        """Lightweight fallback just for calculating Validation Loss over an array."""
        if len(preds) == 0 or len(target_data) == 0:
            return 0.0
        
        min_len = min(len(preds), len(target_data))
        preds_arr = np.array(preds[-min_len:])
        target_arr = target_data[-min_len:]
        
        if preds_arr.ndim == 3:
            preds_arr = preds_arr[:, 0, :]
            
        if hasattr(self.metric, 'metric'):
            return float(self.metric.metric(target_arr, preds_arr))
        return float(np.mean((preds_arr - target_arr) ** 2))

    def save(self, filepath: str):
        with open(filepath, 'wb') as f:
            pickle.dump({'model_state': self.model.get_weights()}, f)

    def load(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Cannot find model file at {filepath}")
        with open(filepath, 'rb') as f:
            state = pickle.load(f)
            weights = state['model_state']
            self.model.mu_tensor = weights['mu']
            self.model.sigma_sq_tensor = weights['sigma']
            self.model.gradient_momentum_tensor = weights['mom']