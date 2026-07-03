import json
import numpy as np
import pandas as pd
import os
from datetime import datetime, timedelta

from data_processor import MarketDataProcessor
from model_engine import ModelEngine 
from reporter import DDSReporter, PriceResultsReporter



def load_config(filepath="config.json"):
    with open(filepath, 'r') as f:
        return json.load(f)

def run_dds_pipeline():
    # 1. Load Configurations
    config = load_config()
    g_params = config["global_params"]
    l_params = config["learning_params"]
    m_params = config["model_params"]
    p_params = config["plot_params"]
    
    # 2. Get Data
    processor = MarketDataProcessor(observation_window=60, params=g_params)
    assets = processor.load_symbols_from_excel(g_params["symbols_file"])
    
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * g_params["history_years"])
    raw_prices = processor.fetch_data(assets, start_date, end_date)
    
    # Updated: Using simple returns directly
    returns_df = processor.compute_returns(raw_prices)
    data_matrix = np.clip(returns_df.values, -10.0, 10.0)
    dates = returns_df.index
    T_total, N = data_matrix.shape
    
    m_params["params"]["n_assets"] = N 

    # 3. Build Model (Passing only learning_params and model_params)
    print("[*] Initializing Pipeline & Model Components...")
    model = ModelEngine(l_params, m_params)

    # 4. Delegate Execution to Model Engine
    print("[*] Handing over data to Model Engine for execution...")
    results = model.execute_pipeline(data_matrix)

    # ---------------------------------------------------------
    # 5. Plotting and Output Generation
    # ---------------------------------------------------------
    print("[*] Launching Output Diagnostics Interface...")
    
    # Generate business dates for the future forecast timeline
    last_real_date = dates[-1]
    future_dates = pd.bdate_range(start=last_real_date + timedelta(days=1), periods=l_params["future_days"])
    extended_dates = dates.append(future_dates)
    
    # 5a. Structural / System Health Diagnostics
    diagnostic_reporter = DDSReporter(model, assets, export_dir=g_params["export_dir"])
    diagnostic_reporter.generate_all_reports(plot_config=p_params)
    
    # 5b. Absolute Price Reconstruction and Plotting (Using unpacked results dict)
    # Removed rolling_means and rolling_stds entirely
    price_reporter = PriceResultsReporter(assets, export_dir=g_params["export_dir"])
    price_reporter.generate_reports(
        dates=extended_dates,
        raw_prices=raw_prices,
        train_preds=results["train_preds"],
        test1_preds=results["test_preds"],
        test2_preds=results["test_preds_autoregressive"],
        future_preds=results["future_preds"],
        train_start=results["warmUp_idx"],
        split_idx=results["split_idx"],
        T_total=T_total,
        plot_config=p_params
    )
    
    # ---------------------------------------------------------
    # 6. Save the Trained Model
    # ---------------------------------------------------------
    model_save_path = os.path.join(g_params["export_dir"], "trained_engine.pkl")
    model.save(model_save_path)
    
    print(f"[*] Pipeline complete. Model successfully saved to {model_save_path}")

if __name__ == "__main__":
    run_dds_pipeline()