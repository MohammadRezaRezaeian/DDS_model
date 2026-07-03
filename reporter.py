import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from forecast import PriceResultsReporter

class DDSReporter:
    def __init__(self, engine, asset_names, export_dir="dds_outputs"):
        self.engine = engine
        self.asset_names = asset_names
        self.export_dir = export_dir
        os.makedirs(self.export_dir, exist_ok=True)

    def generate_all_reports(self, plot_config=None):
        if plot_config is None:
            plot_config = {"save_png": True, "show_plots": False, "export_excel": True}

        print("[*] Extracting Data from Model Engine for Reporting...")
        
        # ---------------------------------------------------------
        # Create DataFrame natively without forcing 'Step'
        # ---------------------------------------------------------
        df_train = pd.DataFrame(self.engine.train_metrics)
        
        # Make the index represent the Epoch (1-indexed for readability)
        if not df_train.empty:
            df_train.index = df_train.index + 1
            df_train.index.name = 'Epoch'
            
        # Safely fetch test_metrics if they exist in older versions, 
        # otherwise use an empty DataFrame so the pipeline doesn't crash
        test_metrics = getattr(self.engine, 'test_metrics', [])
        df_test = pd.DataFrame(test_metrics)
        if not df_test.empty and 'Step' in df_test.columns:
            df_test = df_test.set_index('Step')
        
        # --- Excel Export ---
        if plot_config.get("export_excel", True):
            with pd.ExcelWriter(os.path.join(self.export_dir, "overfitting_metrics.xlsx")) as writer:
                if not df_train.empty:
                    df_train.to_excel(writer, sheet_name="Phase1_Train_Val")
                if not df_test.empty:
                    df_test.to_excel(writer, sheet_name="Phase2_3_Test_Gap")
        
        # --- Plot Generation ---
        save_png = plot_config.get("save_png", True)
        show_plots = plot_config.get("show_plots", False)
        
        if save_png or show_plots:
            # Dynamically check if columns exist before plotting to prevent KeyErrors
            if 'Spectral_Radius' in df_train.columns:
                self._plot_spectral_radius(df_train, save_png, show_plots)

            if 'Stochastic_Spectral_Radius' in df_train.columns:
                self._plot_stochastic_spectral_radius(df_train, save_png, show_plots)
            
            if 'BIC' in df_train.columns:
                self._plot_bic(df_train, save_png, show_plots)
            
            if 'Loss' in df_train.columns:
                self._plot_phase1_holdout(df_train, save_png, show_plots)
                
            if 'Kalman_Trace' in df_train.columns:
                self._plot_kalman_trace(df_train, save_png, show_plots)

            if not df_test.empty and 'Test1_MSE' in df_test.columns:
                self._plot_generalization_gap(df_test, save_png, show_plots)
        
        print(f"[*] Diagnostic Dashboards processed and routed to {self.export_dir}")

    def _handle_plot_output(self, filename, save_png, show_plots):
        plt.tight_layout()
        if save_png:
            plt.savefig(os.path.join(self.export_dir, filename))
        if show_plots:
            plt.show()
        plt.close()

    def _plot_spectral_radius(self, df_train, save_png, show_plots):
        plt.figure(figsize=(12, 6))
        plt.plot(df_train.index, df_train['Spectral_Radius'], color='darkorange', marker='o')
        plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label="OVERFITTING SIGNAL: Chaos Threshold (>1.0)")
        plt.title("Stability Index (Spectral Radius) per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Maximum Eigenvalue")
        plt.legend()
        self._handle_plot_output("metric_1_spectral_radius.png", save_png, show_plots)

    def _plot_stochastic_spectral_radius(self, df_train, save_png, show_plots):
        plt.figure(figsize=(12, 6))
        plt.plot(df_train.index, df_train['Stochastic_Spectral_Radius'], color='darkred', marker='o')
        plt.axhline(1.0, color='red', linestyle='--', linewidth=2, label="OVERFITTING SIGNAL: Chaos Threshold (>1.0)")
        plt.title("Stability Index (Stochastic Spectral Radius) per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("Maximum Eigenvalue")
        plt.legend()
        self._handle_plot_output("metric_5_Stochastic_spectral_radius.png", save_png, show_plots)

    def _plot_generalization_gap(self, df_test, save_png, show_plots):
        plt.figure(figsize=(12, 6))
        plt.plot(df_test.index, df_test['Test1_MSE'], label='Test 1 MSE (Anchored)', color='blue', alpha=0.7)
        plt.plot(df_test.index, df_test['Test2_MSE'], label='Test 2 MSE (Unanchored)', color='magenta', linewidth=2)
        plt.title("Generalization Gap: Structural Integrity")
        plt.ylabel("MSE")
        plt.yscale('log')
        plt.legend()
        self._handle_plot_output("metric_2_generalization_gap.png", save_png, show_plots)

    def _plot_kalman_trace(self, df_train, save_png, show_plots):
        plt.figure(figsize=(12, 6))
        train_only = df_train[df_train['Mode'] == 'Train'] if 'Mode' in df_train.columns else df_train
        
        plt.plot(train_only.index, train_only['Kalman_Trace'], color='purple', marker='o')
        plt.axhline(train_only['Kalman_Trace'].mean(), color='black', linestyle=':', label="Historical Average")
        plt.title("Kalman Gain Trace (Learning Rate)")
        plt.xlabel("Epoch")
        plt.ylabel("Average Gain Matrix Trace")
        plt.legend()
        self._handle_plot_output("metric_3_kalman_trace.png", save_png, show_plots)
        
    def _plot_bic(self, df_train, save_png, show_plots):
        plt.figure(figsize=(12, 6))
        plt.plot(df_train.index, df_train['BIC'], color='green', marker='o')
        plt.title("Bayesian Information Criterion (BIC) per Epoch")
        plt.xlabel("Epoch")
        plt.ylabel("BIC Score")
        self._handle_plot_output("metric_4_bic.png", save_png, show_plots)
        
    def _plot_phase1_holdout(self, df_train, save_png, show_plots):
        plt.figure(figsize=(12, 6))
        plt.plot(df_train.index, df_train['Loss'], marker='o', label='Epoch Training Loss', color='blue')
        plt.title("Phase 1 Epoch Loss Progression")
        plt.xlabel("Epoch")
        plt.ylabel("Loss (MSE)")
        plt.yscale('log')
        plt.legend()
        self._handle_plot_output("metric_0_phase1_holdout.png", save_png, show_plots)


