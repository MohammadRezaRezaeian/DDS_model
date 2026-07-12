import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class PriceResultsReporter:
    def __init__(self, asset_names: list, export_dir: str = "dds_outputs", plot_back_price: int = 200):
        self.asset_names = asset_names
        self.excel_dir = os.path.join(export_dir, "excel_reports")
        self.plot_dir = os.path.join(export_dir, "visual_reports")
        
        # NEW ATTRIBUTE: Limits how far back in time the plots display 
        self.plot_back_price = plot_back_price
        
        os.makedirs(self.excel_dir, exist_ok=True)
        os.makedirs(self.plot_dir, exist_ok=True)

    def _convert_to_real_prices(self, start_idx, end_idx, dates, preds_returns, raw_prices, is_autoregressive=False):
        reconstructed_prices = []
        current_auto_price = None
        
        if is_autoregressive:
            # Prevent wrapping around to the end of the dates array if start_idx=0
            if start_idx > 0:
                prev_date = dates[start_idx - 1]
                current_auto_price = raw_prices.loc[prev_date].values
            else:
                current_auto_price = raw_prices.iloc[0].values

        target_len = end_idx - start_idx
        actual_len = len(preds_returns)
        loop_len = min(target_len, actual_len)

        for i in range(loop_len):
            t = start_idx + i
            
            # Prevent wrapping to the future dates if we are at the absolute beginning of the dataset
            if t == 0:
                reconstructed_prices.append(np.full(len(self.asset_names), np.nan))
                continue
                
            prev_date = dates[t - 1]
            
            # ---------------------------------------------------------
            # Bulletproof Dimensionality Slicing
            # ---------------------------------------------------------
            pred_arr = np.array(preds_returns[i])
            
            # Safely handle padded/empty predictions (Solves the (9,) and (0,) Broadcast Error)
            if pred_arr.size == 0:
                reconstructed_prices.append(np.full(len(self.asset_names), np.nan))
                continue
                
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
            
            # Safely handle empty predictions in the future forecast loop
            if pred_arr.size == 0:
                future_reconstructed.append(np.full(len(self.asset_names), np.nan))
                continue
                
            while pred_arr.ndim > 2:
                pred_arr = pred_arr[0]
            if pred_arr.ndim == 2:
                pred_arr = pred_arr[0]
                
            # NEW MATH: P_t = P_{t-1} * (1 + Simple_Return)
            pred_price = current_future_price * (1.0 + pred_arr)
            current_future_price = pred_price
            future_reconstructed.append(pred_price)

        # FIX: Align Prediction Length with Calendar Date Length
        future_index = dates[T_total:]
        actual_future_len = min(len(future_reconstructed), len(future_index))
        
        df_pred_future = pd.DataFrame(
            future_reconstructed[:actual_future_len], 
            index=future_index[:actual_future_len], 
            columns=self.asset_names
        )

        if plot_config.get("export_excel", True):
            excel_path = os.path.join(self.excel_dir, "absolute_price_predictions.xlsx")
            with pd.ExcelWriter(excel_path) as writer:
                df_actual_test.to_excel(writer, sheet_name="Actual_Prices_Test_Phase")
                if not df_pred_test1.empty:
                    df_pred_test1.to_excel(writer, sheet_name="Test1_OneStep_Preds")
                if not df_pred_test2.empty:
                    df_pred_test2.to_excel(writer, sheet_name="Test2_MultiStep_Preds")
                if not df_pred_future.empty:
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
            
            # ---------------------------------------------------------
            # PLOT WINDOW SLICING LOGIC
            # ---------------------------------------------------------
            if self.plot_back_price and self.plot_back_price > 0:
                # Find the date corresponding to `plot_back_price` steps ago
                plot_start_idx = max(0, T_total - self.plot_back_price)
                plot_start_date = dates[plot_start_idx]
            else:
                # If set to 0 or None, plot all historical data
                plot_start_date = df_actual_full.index[0]
                
            for asset in self.asset_names:
                plt.figure(figsize=(14, 7))
                
                # Slice the Actual Market Prices
                plot_actual = df_actual_full.loc[plot_start_date:]
                plt.plot(plot_actual.index, plot_actual[asset], label="Actual Market Price", color='black', linewidth=1.5, zorder=1)
                
                # Slice the Train predictions
                if not df_pred_test1.empty and asset in df_pred_test1.columns:
                    test1_y = df_pred_test1[asset].loc[plot_start_date:].copy()
                    if not test1_y.empty:
                        valid_idx = test1_y.index.intersection(df_actual_test.index)
                        test1_pct_dev = np.abs(test1_y.loc[valid_idx] - df_actual_test[asset].loc[valid_idx]) / df_actual_test[asset].loc[valid_idx]
                        
                        # --- THE FIX ---
                        # 1. Find the exact dates where the outlier condition is True
                        test1_outlier_dates = valid_idx[test1_pct_dev > outlier_threshold]
                        # 2. Assign NaN in a single, unchained step
                        test1_y.loc[test1_outlier_dates] = np.nan 
                        
                        plt.plot(test1_y.index, test1_y, label="Test 1: One-Step Ahead", color='cyan', alpha=0.9, marker='x', markersize=4, linestyle='None', zorder=3)
                
                # Slice the Test 1 predictions
                if not df_pred_test1.empty and asset in df_pred_test1.columns:
                    test1_y = df_pred_test1[asset].loc[plot_start_date:].copy()
                    if not test1_y.empty:
                        valid_idx = test1_y.index.intersection(df_actual_test.index)
                        test1_pct_dev = np.abs(test1_y.loc[valid_idx] - df_actual_test[asset].loc[valid_idx]) / df_actual_test[asset].loc[valid_idx]
                        outlier_dates = valid_idx[test1_pct_dev > outlier_threshold]
                        test1_y.loc[outlier_dates] = np.nan
                        plt.plot(test1_y.index, test1_y, label="Test 1: One-Step Ahead", color='cyan', alpha=0.9, marker='x', markersize=4, linestyle='None', zorder=3)

                # Slice the Test 2 predictions
                if not df_pred_test2.empty and asset in df_pred_test2.columns:
                    test2_y = df_pred_test2[asset].loc[plot_start_date:]
                    if not test2_y.empty:
                        plt.plot(test2_y.index, test2_y, label="Test 2: Autoregressive Drift", color='magenta', linewidth=2.0, zorder=4)

                # Future predictions natively occur AFTER the slice window, so we plot them fully
                if not df_pred_future.empty and asset in df_pred_future.columns:
                    future_y = df_pred_future[asset]
                    plt.plot(future_y.index, future_y, label="Phase 4: Pure Future Forecast", color='gold', linewidth=2.5, linestyle='-', zorder=5)

                # Only draw the Train/Test line if it occurs inside our zoomed-in plot window
                split_date = dates[split_idx]
                if split_date >= plot_start_date:
                    plt.axvline(x=split_date, color='gray', linestyle='--', label='Train/Test Split')
                    
                present_date = dates[T_total-1]
                if present_date >= plot_start_date:
                    plt.axvline(x=present_date, color='red', linestyle='-', linewidth=2, label='Present Day (End of Data)')

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