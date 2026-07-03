import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime
import pytz
import os

class MarketDataProcessor:
    def __init__(self, observation_window=60, params: dict = None):
        self.observation_window = observation_window
        self.source = params.get("source", "meta").lower()
        self.mt5_path = params.get("mt5_path", "C:\\Program Files\\WM Markets MT5 Terminal\\terminal64.exe")
        self.csv_path = "./data/data.csv"
        self.rolling_means = {}
        self.rolling_stds = {}

    def load_symbols_from_excel(self, file_path="./model/symbols.xlsx", column_name="Symbol") -> list:
        """
        Reads the asset symbols from an Excel file.
        Assumes the symbols are listed under a specific column header (default: 'Symbol').
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Cannot find the Excel file at {file_path}")
            
        print(f"[*] Reading symbols from {file_path}...")
        df = pd.read_excel(file_path)
        
        if column_name not in df.columns:
            # Fallback to the first column if the specified name isn't found
            symbols = df.iloc[:, 0].dropna().astype(str).tolist()
        else:
            symbols = df[column_name].dropna().astype(str).tolist()
            
        # Clean up any whitespace
        symbols = [s.strip() for s in symbols]
        print(f"[*] Loaded {len(symbols)} symbols: {symbols}")
        return symbols

    def fetch_data(self, symbols: list, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        Master router method that fetches data either from MT5 or a local CSV.
        """
        if self.source == "meta":
            return self.fetch_mt5_data(symbols, start_date, end_date)
        elif self.source == "csv":
            if not self.csv_path:
                raise ValueError("csv_path must be provided in config when source is 'csv'")
            return self.fetch_csv_data(symbols, self.csv_path)
        else:
            raise ValueError(f"Unknown data source: '{self.source}'. Valid options are 'meta' or 'csv'.")

    def fetch_csv_data(self, symbols: list, csv_path: str) -> pd.DataFrame:
        """
        Reads historical close prices from a local CSV file.
        Assumes the first column contains Datetimes, and column headers are symbol names.
        """
        print(f"[*] Initializing CSV data extraction from {csv_path}...")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Cannot find the CSV file at {csv_path}")
            
        # Read the CSV, enforcing the first column (0) as the Datetime index
        df = pd.read_csv(csv_path, parse_dates=[0], index_col=0)
        df.index.name = 'time'
        
        # Filter for requested symbols that actually exist in the CSV
        available_symbols = [s for s in symbols if s in df.columns]
        missing_symbols = [s for s in symbols if s not in df.columns]
        
        if missing_symbols:
            print(f"[!] Warning: The following symbols were not found in the CSV and will be skipped: {missing_symbols}")
            
        if not available_symbols:
            raise ValueError("No matching symbols found in the provided CSV file.")

        # Extract only the required columns and forward-fill missing data
        raw_prices_df = df[available_symbols].copy()
        raw_prices_df.ffill(inplace=True)
        
        print(f"[*] Successfully built raw price matrix from CSV with shape {raw_prices_df.shape}")
        return raw_prices_df

    def fetch_mt5_data(self, symbols: list, start_date: datetime, end_date: datetime, timeframe=mt5.TIMEFRAME_D1) -> pd.DataFrame:
        """
        Connects to the specified MT5 terminal, enables the symbols, and fetches historical close prices.
        """
        print(f"[*] Initializing MetaTrader 5 from {self.mt5_path}...")
        
        # Initialize MT5 connection targeting the specific terminal
        if not mt5.initialize(path=self.mt5_path):
            error_code = mt5.last_error()
            mt5.shutdown()
            raise ConnectionError(f"MT5 initialization failed. Error code: {error_code}")
            
        # Set timezone to UTC to align with MT5 server time safely
        timezone = pytz.timezone("Etc/UTC")
        start_date = start_date.replace(tzinfo=timezone) if start_date.tzinfo is None else start_date
        end_date = end_date.replace(tzinfo=timezone) if end_date.tzinfo is None else end_date

        price_data = {}

        for symbol in symbols:
            # Ensure symbol is visible in Market Watch
            selected = mt5.symbol_select(symbol, True)
            if not selected:
                print(f"[!] Warning: Failed to select '{symbol}'. Skipping.")
                continue
                
            # Fetch data
            rates = mt5.copy_rates_range(symbol, timeframe, start_date, end_date)
            
            if rates is None or len(rates) == 0:
                print(f"[!] Warning: No data retrieved for '{symbol}'.")
                continue
                
            # Convert to DataFrame
            df_rates = pd.DataFrame(rates)
            df_rates['time'] = pd.to_datetime(df_rates['time'], unit='s')
            
            # We use the 'close' price for our DDS model
            price_data[symbol] = df_rates.set_index('time')['close']

        # Shut down MT5 connection
        mt5.shutdown()
        
        if not price_data:
            raise ValueError("No data could be fetched for any of the provided symbols.")

        # Combine all series into a single DataFrame, forward-filling missing days
        raw_prices_df = pd.DataFrame(price_data)
        raw_prices_df.ffill(inplace=True)
        
        print(f"[*] Successfully built raw price matrix with shape {raw_prices_df.shape}")
        return raw_prices_df

    def compute_returns(self, prices_df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts raw prices to log returns: r_{i,t} = ln(P_{i,t} / P_{i,t-1})
        """
        prices_df = prices_df.ffill()
        prevPrices = prices_df.shift(1)
        log_returns = ((prices_df -prevPrices) / prevPrices).fillna(0)
        return log_returns

    def standardize_returns(self, log_returns_df: pd.DataFrame) -> pd.DataFrame:
        standardized = pd.DataFrame(index=log_returns_df.index, columns=log_returns_df.columns)
        
        # INCREASE THE VOLATILITY FLOOR
        # Change it from 1e-8 to a realistic minimum daily volatility (e.g., 0.001 or 0.1%)
        MIN_VOLATILITY = 0.001 
        
        for t in range(self.observation_window, len(log_returns_df)):
            window = log_returns_df.iloc[t - self.observation_window : t]
            mu = window.mean()
            sigma = window.std()
            
            self.rolling_means[log_returns_df.index[t]] = mu
            self.rolling_stds[log_returns_df.index[t]] = sigma
            
            # Use np.maximum to ensure sigma never drops below the floor
            safe_sigma = np.maximum(sigma, MIN_VOLATILITY)
            standardized.iloc[t] = (log_returns_df.iloc[t] - mu) / safe_sigma
            
        return standardized.dropna().astype(float)