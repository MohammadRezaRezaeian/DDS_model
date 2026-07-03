import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PriceResultsReporter:
    def __init__(self, asset_names: list, export_dir: str = "dds_outputs"):
        self.asset_names = asset_names
        self.excel_dir = os.path.join(export_dir, "excel_reports")
        self.plot_dir = os.path.join(export_dir, "visual_reports")
        os.makedirs(self.excel_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def _convert_to_real_prices(self, start_idx, end_idx, dates, preds_returns, raw_prices, is_autoregressive=False):
        reconstructed_prices = []
        current_auto_price = None
        
        if is_autoregressive:
            prev_date = dates[start_idx - 1]
            current_auto_price = raw_prices.loc[prev_date].values

        target_len = end_idx - start_idx
        actual_len = len(preds_returns)
        loop_len = min(target_len, actual_len)

        for i in range(loop_len):
            t = start_idx + i
            prev_date = dates[t - 1]
            
            # ---------------------------------------------------------
            # Bulletproof Dimensionality Slicing
            # ---------------------------------------------------------
            pred_arr = np.array(preds_returns[i])
            
            while pred_arr.ndim > 2:
                pred_arr = pred_arr[0]
                
            if pred_arr.ndim == 2:
                pred_arr = pred_arr[0] 
                
            # NEW MATH: P_t = P_{t-1} * (1 + Simple_Return)
            if is_autoregressive:
                pred_price = current_auto_price * (1.0 + pred_arr)
                current_auto_price = pred_price 
            else:
                prev_real_price = raw_prices.loc[prev_date].values
                pred_price = prev_real_price * (1.0 + pred_arr)
                
            reconstructed_prices.append(pred_price)

        return pd.DataFrame(reconstructed_prices, index=dates[start_idx : start_idx + loop_len], columns=self.asset_names)

    def generate_reports(self, dates, raw_prices, train_preds, test1_preds, test2_preds, 
                         future_preds, train_start, split_idx, T_total, outlier_threshold=0.08, plot_config=None):
        
        if plot_config is None:
            plot_config = {"save_png": True, "show_plots": False, "export_excel": True}
            
        print("[*] Reconstructing Absolute Prices for Financial Reporting...")
        
        df_pred_train = self._convert_to_real_prices(
            train_start, split_idx, dates, train_preds, raw_prices
        )
        
        df_pred_test1 = self._convert_to_real_prices(
            split_idx, T_total, dates, test1_preds, raw_prices, is_autoregressive=False
        )
        
        df_pred_test2 = self._convert_to_real_prices(
            split_idx, T_total, dates, test2_preds, raw_prices, is_autoregressive=True
        )

        df_actual_full = raw_prices.loc[dates[train_start]:dates[T_total-1]]
        df_actual_test = raw_prices.loc[dates[split_idx]:dates[T_total-1]]

        future_reconstructed = []
        last_date = dates[T_total - 1]
        last_real_price = raw_prices.loc[last_date].values
        
        current_future_price = last_real_price

        for pred_ret in future_preds:
            # Apply identical bulletproof slicing to the future loop
            pred_arr = np.array(pred_ret)
            while pred_arr.ndim > 2:
                pred_arr = pred_arr[0]
            if pred_arr.ndim == 2:
                pred_arr = pred_arr[0]
                
            # NEW MATH: P_t = P_{t-1} * (1 + Simple_Return)
            pred_price = current_future_price * (1.0 + pred_arr)
            current_future_price = pred_price
            future_reconstructed.append(pred_price)

        df_pred_future = pd.DataFrame(future_reconstructed, index=dates[T_total:], columns=self.asset_names)

        if plot_config.get("export_excel", True):
            excel_path = os.path.join(self.excel_dir, "absolute_price_predictions.xlsx")
            with pd.ExcelWriter(excel_path) as writer:
                df_actual_test.to_excel(writer, sheet_name="Actual_Prices_Test_Phase")
                if not df_pred_test1.empty:
                    df_pred_test1.to_excel(writer, sheet_name="Test1_OneStep_Preds")
                if not df_pred_test2.empty:
                    df_pred_test2.to_excel(writer, sheet_name="Test2_MultiStep_Preds")
                df_pred_future.to_excel(writer, sheet_name="Phase4_Future_Forecast")
                
                aligned_test1, aligned_actual1 = df_pred_test1.align(df_actual_test, join='inner')
                aligned_test2, aligned_actual2 = df_pred_test2.align(df_actual_test, join='inner')
                
                error_test1 = aligned_test1 - aligned_actual1
                error_test2 = aligned_test2 - aligned_actual2
                error_test1.to_excel(writer, sheet_name="Test1_Dollar_Error")
                error_test2.to_excel(writer, sheet_name="Test2_Dollar_Error")

        save_png = plot_config.get("save_png", True)
        show_plots = plot_config.get("show_plots", False)

        if save_png or show_plots:
            for asset in self.asset_names:
                plt.figure(figsize=(14, 7))
                
                plt.plot(df_actual_full.index, df_actual_full[asset], label="Actual Market Price", color='black', linewidth=1.5, zorder=1)
                
                if not df_pred_train.empty and asset in df_pred_train.columns:
                    train_y = df_pred_train[asset].copy()
                    valid_idx = train_y.index.intersection(df_actual_full.index)
                    train_pct_dev = np.abs(train_y.loc[valid_idx] - df_actual_full[asset].loc[valid_idx]) / df_actual_full[asset].loc[valid_idx]
                    train_y.loc[valid_idx][train_pct_dev > outlier_threshold] = np.nan 
                    plt.plot(train_y.index, train_y, label="Train Phase (Learning)", color='blue', alpha=0.3, marker='.', markersize=3, linestyle='None', zorder=2)
                
                if not df_pred_test1.empty and asset in df_pred_test1.columns:
                    test1_y = df_pred_test1[asset].copy()
                    valid_idx = test1_y.index.intersection(df_actual_test.index)
                    test1_pct_dev = np.abs(test1_y.loc[valid_idx] - df_actual_test[asset].loc[valid_idx]) / df_actual_test[asset].loc[valid_idx]
                    test1_y.loc[valid_idx][test1_pct_dev > outlier_threshold] = np.nan 
                    plt.plot(test1_y.index, test1_y, label="Test 1: One-Step Ahead", color='cyan', alpha=0.9, marker='x', markersize=4, linestyle='None', zorder=3)

                if not df_pred_test2.empty and asset in df_pred_test2.columns:
                    test2_y = df_pred_test2[asset]
                    plt.plot(test2_y.index, test2_y, label="Test 2: Autoregressive Drift", color='magenta', linewidth=2.0, zorder=4)

                if not df_pred_future.empty and asset in df_pred_future.columns:
                    future_y = df_pred_future[asset]
                    plt.plot(future_y.index, future_y, label="Phase 4: Pure Future Forecast", color='gold', linewidth=2.5, linestyle='-', zorder=5)

                plt.axvline(x=dates[split_idx], color='gray', linestyle='--', label='Train/Test Split')
                plt.axvline(x=dates[T_total-1], color='red', linestyle='-', linewidth=2, label='Present Day (End of Data)')

                plt.title(f"Absolute Price Prediction Engine: {asset}\n(Gold line is completely blind future projection)")
                plt.ylabel("Price ($)")
                plt.grid(True, alpha=0.2)
                plt.legend()
                plt.tight_layout()
                
                if save_png:
                    filename = "".join([c for c in asset if c.isalpha() or c.isdigit()]).rstrip()
                    plt.savefig(os.path.join(self.plot_dir, f"price_prediction_{filename}.png"))
                if show_plots:
                    plt.show()
                plt.close()