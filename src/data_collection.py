"""
Data collection module for Phillips Curve ML project.
Fetches economic data from FRED API for multiple countries (2015-2025).
"""

import pandas as pd
import numpy as np
from fredapi import Fred
from datetime import datetime
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')


class FREDDataCollector:
    """
    Collects economic data from FRED API for Phillips Curve analysis.
    Handles 2015-2025 time period with proper data alignment.
    """
    
    def __init__(self, api_key: str):
        """
        Initialize FRED API connection.
        
        Parameters:
        -----------
        api_key : str
            Your FRED API key from https://fred.stlouisfed.org
        """
        self.fred = Fred(api_key=api_key)
        
        # Define series IDs for US and UK (2015-2025)
        self.series_ids = {
            'US': {
                'unemployment': 'UNRATE',              # Unemployment Rate (%)
                'inflation': 'CPALTT01USM659N',        # CPI YoY inflation (ALREADY %)
                'policy_rate': 'FEDFUNDS'              # Federal Funds Effective Rate (%)
            },
            'UK': {
                'unemployment': 'LRHUTTTTGBM156S',     # Unemployment Rate Total (%)
                'inflation': 'GBRCPALTT01CTGYM',       # CPI YoY inflation (ALREADY %)
                'policy_rate': 'IRSTCI01GBM156N'       # Bank Rate - Immediate Rates (%)
            }
        }
        
        self.countries = list(self.series_ids.keys())
    
    def fetch_series(self, series_id: str, start_date: str, end_date: str) -> pd.Series:
        """
        Fetch a single time series from FRED.
        
        Parameters:
        -----------
        series_id : str
            FRED series identifier
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pd.Series
            Time series data or empty Series if error
        """
        try:
            data = self.fred.get_series(
                series_id, 
                observation_start=start_date,
                observation_end=end_date
            )
            return data
        except Exception as e:
            print(f"  ⚠️  Error fetching {series_id}: {e}")
            return pd.Series(dtype=float)
    
    def fetch_country_data(self, country: str, start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch all economic indicators for a specific country.
        
        Parameters:
        -----------
        country : str
            Country code ('US' or 'UK')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with all indicators for country
        """
        if country not in self.series_ids:
            raise ValueError(f"Country {country} not supported")
        
        print(f"  Fetching {country} data...")
        
        series_dict = self.series_ids[country]
        data = {}
        
        for indicator, series_id in series_dict.items():
            print(f"    - {indicator} ({series_id})")
            data[indicator] = self.fetch_series(series_id, start_date, end_date)
        
        df = pd.DataFrame(data)
        df['country'] = country
        
        return df
    
    def fetch_all_countries(self, start_date: str = '2015-01-01', 
                           end_date: str = None) -> pd.DataFrame:
        """
        Fetch data for all countries and combine.
        
        Parameters:
        -----------
        start_date : str
            Start date (default: 2015-01-01)
        end_date : str, optional
            End date (defaults to today)
            
        Returns:
        --------
        pd.DataFrame
            Combined dataset with all countries
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
        
        print(f"\nFetching FRED data ({start_date} to {end_date})...")
        
        all_data = []
        for country in self.countries:
            country_df = self.fetch_country_data(country, start_date, end_date)
            all_data.append(country_df)
        
        combined_df = pd.concat(all_data, axis=0)
        combined_df = combined_df.reset_index()
        combined_df = combined_df.rename(columns={'index': 'date'})
        
        return combined_df
    
    def clean_and_filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data and align to common date range.
        Ensures both countries have complete data for fair comparison.
        """
        print("\n" + "="*70)
        print("DATA QUALITY ASSESSMENT & ALIGNMENT")
        print("="*70)
        
        df = df.sort_values(['country', 'date']).reset_index(drop=True)
        
        # Step 1: Find data availability
        print("\nData Availability by Country and Variable:")
        print("-"*70)
        
        for country in sorted(df['country'].unique()):
            country_data = df[df['country'] == country]
            print(f"\n{country}:")
            
            for col in ['unemployment', 'inflation', 'policy_rate']:
                last_valid = country_data[country_data[col].notna()]['date'].max()
                first_valid = country_data[country_data[col].notna()]['date'].min()
                coverage = country_data[col].notna().sum() / len(country_data) * 100
                
                print(f"  {col:15s}: {first_valid.date()} to {last_valid.date()} ({coverage:.1f}% coverage)")
        
        # Step 2: Find common end date
        print("\n" + "-"*70)
        print("Finding Common Data Window...")
        print("-"*70)
        
        latest_dates = {}
        for country in df['country'].unique():
            country_data = df[df['country'] == country]
            complete = country_data.dropna(subset=['unemployment', 'inflation', 'policy_rate'])
            
            if len(complete) > 0:
                latest_date = complete['date'].max()
                latest_dates[country] = latest_date
                print(f"\n{country}: Last complete observation = {latest_date.date()}")
        
        common_end_date = min(latest_dates.values())
        print(f"\n{'='*70}")
        print(f"OPTIMAL COMMON END DATE: {common_end_date.date()}")
        print(f"{'='*70}\n")
        
        # Step 3: Filter and drop NaN
        df_aligned = df[df['date'] <= common_end_date].copy()
        df_clean = df_aligned.dropna(subset=['unemployment', 'inflation', 'policy_rate'])
        
        print("Final Dataset After Alignment:")
        print("-"*70)
        
        for country in sorted(df_clean['country'].unique()):
            country_clean = df_clean[df_clean['country'] == country]
            print(f"\n{country}:")
            print(f"  Observations: {len(country_clean)}")
            print(f"  Date range: {country_clean['date'].min().date()} to {country_clean['date'].max().date()}")
            print(f"  Years: {(country_clean['date'].max() - country_clean['date'].min()).days / 365.25:.1f}")
        
        print(f"\n{'-'*70}")
        print(f"TOTAL DATASET: {len(df_clean)} rows (aligned period)")
        print(f"GLOBAL DATE RANGE: {df_clean['date'].min().date()} to {df_clean['date'].max().date()}")
        print("="*70 + "\n")
        
        return df_clean
    
    def diagnose_missing_data(self, df: pd.DataFrame) -> None:
        """
        Show detailed missing data analysis before and after alignment.
        """
        print("\n" + "="*70)
        print("DETAILED MISSING DATA ANALYSIS")
        print("="*70)
        
        for country in sorted(df['country'].unique()):
            country_data = df[df['country'] == country].sort_values('date')
            
            print(f"\n{country} ({len(country_data)} rows):")
            print(f"  Date range: {country_data['date'].min().date()} to {country_data['date'].max().date()}")
            
            for col in ['unemployment', 'inflation', 'policy_rate']:
                missing_mask = country_data[col].isna()
                n_missing = missing_mask.sum()
                
                if n_missing > 0:
                    missing_rows = country_data[missing_mask]
                    print(f"\n  {col}: {n_missing} missing values")
                    print(f"    First missing: {missing_rows['date'].iloc[0].date()}")
                    print(f"    Last missing:  {missing_rows['date'].iloc[-1].date()}")
                else:
                    print(f"  {col}: ✓ Complete")

    def save_data(self, df: pd.DataFrame, filename: str, subfolder: str = 'raw') -> str:
        """Save data to CSV file."""
        output_path = os.path.join('data', subfolder, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  ✓ Data saved to: {output_path}")
        return output_path


def main():
    """Main execution: collect, clean, and save data."""
    load_dotenv()
    API_KEY = os.getenv('FRED_API_KEY')
    
    if API_KEY is None:
        raise ValueError(
            "FRED_API_KEY not found in .env file. "
            "Create .env file in project root with: FRED_API_KEY=your_key_here"
        )
    
    print("\n" + "="*70)
    print("PHILLIPS CURVE ML PROJECT - DATA COLLECTION (2015-2025)")
    print("="*70)
    print("\n✓ Initializing FRED API connection...")
    
    collector = FREDDataCollector(api_key=API_KEY)
    
    print("\n✓ Fetching data from FRED...")
    df_raw = collector.fetch_all_countries(start_date='2015-01-01')
    
    print("\n✓ Cleaning and aligning data...")
    df_clean = collector.clean_and_filter_data(df_raw)
    
    print("\n✓ Saving data...")
    collector.save_data(df_raw, 'raw_data.csv', subfolder='raw')
    collector.save_data(df_clean, 'processed_data.csv', subfolder='processed')
    
    print("\n" + "="*70)
    print("DATA COLLECTION COMPLETE!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Review: data/processed/processed_data.csv")
    print(f"  2. Run: Feature engineering (feature_engineering.py)")
    print(f"  3. Run: ML models training (models_country_comparison.py)")
    print("\n")
    
    return df_clean


if __name__ == "__main__":
    df = main()
