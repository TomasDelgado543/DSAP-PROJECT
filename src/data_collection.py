"""
Data collection module for Phillips Curve ML project.
Fetches economic data from FRED API for US and UK (2018-2025).
"""

import pandas as pd
from fredapi import Fred
from datetime import datetime
import os
from dotenv import load_dotenv
import warnings

warnings.filterwarnings('ignore')


class FREDDataCollector:
    """
    Collects economic data from FRED API for Phillips Curve ML analysis.
    Focuses on US and UK with complete data coverage (2018-2025).
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
        
        # Define series IDs for US and UK (optimized for data completeness)
        self.series_ids = {
            'US': {
                'unemployment': 'UNRATE',              # Unemployment Rate (%)
                'inflation': 'CPIAUCSL',               # CPI for All Urban Consumers
                'policy_rate': 'FEDFUNDS'              # Federal Funds Effective Rate (%)
            },
            'UK': {
                'unemployment': 'LRHUTTTTGBM156S',     # Unemployment Rate Total (%)
                'inflation': 'GBRCPALTT01CTGYM',       # Consumer Price Index Total (YoY %)
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
            FRED series identifier (e.g., 'UNRATE')
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
            
        Returns:
        --------
        pd.Series
            Time series data with DatetimeIndex, or empty Series if error
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
            DataFrame with columns: unemployment, inflation, policy_rate, country
        """
        if country not in self.series_ids:
            raise ValueError(f"Country {country} not supported. Available: {self.countries}")
        
        print(f"  Fetching {country} data...")
        
        series_dict = self.series_ids[country]
        data = {}
        
        # Fetch each indicator
        for indicator, series_id in series_dict.items():
            print(f"    - {indicator} ({series_id})")
            data[indicator] = self.fetch_series(series_id, start_date, end_date)
        
        # Combine into DataFrame
        df = pd.DataFrame(data)
        df['country'] = country
        
        return df
    
    def fetch_all_countries(self, start_date: str = '2018-01-01', 
                           end_date: str = None) -> pd.DataFrame:
        """
        Fetch data for all countries and combine into single DataFrame.
        
        Parameters:
        -----------
        start_date : str
            Start date in 'YYYY-MM-DD' format (default: 2018-01-01)
        end_date : str, optional
            End date in 'YYYY-MM-DD' format (defaults to today)
            
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
        
        # Combine all countries
        combined_df = pd.concat(all_data, axis=0)
        combined_df = combined_df.reset_index()
        combined_df = combined_df.rename(columns={'index': 'date'})
        
        return combined_df
    
    def clean_and_filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data and remove rows with missing values (list-wise deletion).
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw data from FRED
        
        Returns:
        --------
        pd.DataFrame
            Cleaned dataset with complete cases only
        """
        print("\n" + "="*70)
        print("DATA QUALITY ASSESSMENT")
        print("="*70)
        
        # Sort by country and date
        df = df.sort_values(['country', 'date']).reset_index(drop=True)
        
        # Display raw data statistics
        print("\nRaw Data Coverage:")
        for country in sorted(df['country'].unique()):
            country_data = df[df['country'] == country]
            cov_unemp = country_data['unemployment'].notna().sum() / len(country_data) * 100
            cov_infl = country_data['inflation'].notna().sum() / len(country_data) * 100
            cov_rate = country_data['policy_rate'].notna().sum() / len(country_data) * 100
            
            print(f"\n  {country}:")
            print(f"    Rows: {len(country_data)}")
            print(f"    Unemployment coverage: {cov_unemp:.1f}%")
            print(f"    Inflation coverage: {cov_infl:.1f}%")
            print(f"    Policy Rate coverage: {cov_rate:.1f}%")
        
        # List-wise deletion: drop rows where ANY variable is missing
        df_clean = df.dropna(subset=['unemployment', 'inflation', 'policy_rate'])
        
        # Display cleaned data statistics
        print("\n" + "-"*70)
        print("After Removing Rows with Missing Values:")
        for country in sorted(df_clean['country'].unique()):
            country_clean = df_clean[df_clean['country'] == country]
            print(f"\n  {country}:")
            print(f"    Valid rows: {len(country_clean)}")
            print(f"    Date range: {country_clean['date'].min().date()} to {country_clean['date'].max().date()}")
            print(f"    Complete observations: {len(country_clean)} months")
        
        print(f"\n{'-'*70}")
        print(f"FINAL DATASET: {len(df_clean)} rows across {df_clean['country'].nunique()} countries")
        print("="*70 + "\n")
        
        return df_clean
    
    def save_data(self, df: pd.DataFrame, filename: str, subfolder: str = 'raw') -> str:
        """
        Save data to CSV file.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to save
        filename : str
            Output filename (e.g., 'raw_data.csv')
        subfolder : str
            Subfolder within data/ (default: 'raw')
        
        Returns:
        --------
        str
            Path where data was saved
        """
        output_path = os.path.join('data', subfolder, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  ✓ Data saved to: {output_path}")
        return output_path
    
    def calculate_inflation_rate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate year-over-year inflation rate from CPI index.
        Converts CPI levels to percentage changes.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame with inflation as CPI index
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with inflation as YoY percentage change
        """
        df = df.copy()
        
        for country in df['country'].unique():
            country_mask = df['country'] == country
            # Convert CPI index to YoY % change
            df.loc[country_mask, 'inflation'] = df.loc[country_mask, 'inflation'].pct_change(12) * 100
        
        return df


def main():
    """
    Main execution: collect, clean, and save data.
    """
    # Load environment variables from .env file
    load_dotenv()
    
    # Get API key from environment
    API_KEY = os.getenv('FRED_API_KEY')
    
    if API_KEY is None:
        raise ValueError(
            "FRED_API_KEY not found in .env file. "
            "Create .env file in project root with: FRED_API_KEY=your_key_here"
        )
    
    # Initialize collector
    print("\n" + "="*70)
    print("PHILLIPS CURVE ML PROJECT - DATA COLLECTION")
    print("="*70)
    print("\n✓ Initializing FRED API connection...")
    
    collector = FREDDataCollector(api_key=API_KEY)
    
    # Fetch raw data (2018-2025)
    print("\n✓ Fetching data from FRED...")
    df_raw = collector.fetch_all_countries(start_date='2018-01-01')
    
    # Clean data (remove rows with any missing values)
    print("\n✓ Cleaning data...")
    df_clean = collector.clean_and_filter_data(df_raw)
    
    # Calculate inflation rate from CPI index
    print("\n✓ Converting CPI index to YoY inflation rate...")
    df_clean = collector.calculate_inflation_rate(df_clean)
    
    # Save both versions
    print("\n✓ Saving data...")
    collector.save_data(df_raw, 'raw_data.csv', subfolder='raw')
    collector.save_data(df_clean, 'processed_data.csv', subfolder='processed')
    
    print("\n" + "="*70)
    print("DATA COLLECTION COMPLETE!")
    print("="*70)
    print(f"\nNext steps:")
    print(f"  1. Review: data/processed/processed_data.csv")
    print(f"  2. Run: Feature engineering (feature_engineering.py)")
    print(f"  3. Run: ML models training (models.py)")
    print("\n")
    
    return df_clean


if __name__ == "__main__":
    df = main()
