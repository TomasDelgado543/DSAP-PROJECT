"""
Data collection module for Phillips Curve ML project.
Loads pre-downloaded data from CSV files instead of FRED API.
Teachers version to run the code without an API key.
"""

import pandas as pd
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')


class LocalDataLoader:
    """
    Loads and aligns economic data from local CSV files.
    No API key required - just raw data files.
    """
    
    def __init__(self, raw_data_folder='data/raw'):
        """
        Initialize loader.
        
        Parameters:
        -----------
        raw_data_folder : str
            Folder containing raw data CSV files
        """
        self.raw_data_folder = raw_data_folder
        self.countries = ['US', 'UK']
    
    def load_country_data(self, filepath: str) -> pd.DataFrame:
        """
        Load data from CSV file.
        
        Parameters:
        -----------
        filepath : str
            Path to CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Data file not found: {filepath}")
        
        print(f"  Loading: {filepath}")
        df = pd.read_csv(filepath)
        
        # Ensure date column is datetime
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        
        return df
    
    def load_all_countries(self) -> pd.DataFrame:
        """
        Load data for all countries from local CSV files.
        
        Expected file structure:
        - data/raw/raw_data.csv (from FRED API download)
        OR
        - Individual CSV files with country data
        
        Returns:
        --------
        pd.DataFrame
            Combined dataset with all countries
        """
        print(f"\nLoading local data from: {self.raw_data_folder}/")
        
        # Try loading combined file first
        combined_path = os.path.join(self.raw_data_folder, 'raw_data.csv')
        
        if os.path.exists(combined_path):
            print(f"\n✓ Found combined data file: raw_data.csv")
            df = self.load_country_data(combined_path)
            
            if 'date' in df.columns:
                df['date'] = pd.to_datetime(df['date'])
            
            return df
        
        # Otherwise, try loading individual country files
        print(f"⚠️  Combined file not found. Looking for individual country files...")
        
        all_data = []
        for country in self.countries:
            country_file = os.path.join(self.raw_data_folder, f'{country.lower()}_data.csv')
            
            if os.path.exists(country_file):
                print(f"\n  ✓ Found {country} file: {country_file}")
                df_country = self.load_country_data(country_file)
                df_country['country'] = country
                all_data.append(df_country)
            else:
                print(f"  ⚠️  {country} file not found: {country_file}")
        
        if not all_data:
            raise FileNotFoundError(
                f"No data files found in {self.raw_data_folder}/\n"
                f"Expected either:\n"
                f"  - data/raw/raw_data.csv (combined), OR\n"
                f"  - data/raw/us_data.csv + data/raw/uk_data.csv (individual)"
            )
        
        combined_df = pd.concat(all_data, axis=0)
        return combined_df
    
    def clean_and_filter_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean data and align to common date range.
        Ensures both countries have complete data for fair comparison.
        """
        print("\n" + "="*70)
        print("DATA QUALITY ASSESSMENT & ALIGNMENT")
        print("="*70)
        
        # Ensure proper data types
        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values(['country', 'date']).reset_index(drop=True)
        
        # Step 1: Find data availability
        print("\nData Availability by Country and Variable:")
        print("-"*70)
        
        for country in sorted(df['country'].unique()):
            country_data = df[df['country'] == country]
            print(f"\n{country}:")
            
            for col in ['unemployment', 'inflation', 'policy_rate']:
                if col in country_data.columns:
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
    
    def save_data(self, df: pd.DataFrame, filename: str, subfolder: str = 'processed') -> str:
        """Save data to CSV file."""
        output_path = os.path.join('data', subfolder, filename)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"  ✓ Data saved to: {output_path}")
        return output_path


def main():
    """Main execution: load, clean, and save data from local CSV files."""
    
    print("\n" + "="*70)
    print("PHILLIPS CURVE ML PROJECT - DATA COLLECTION (LOCAL CSV VERSION)")
    print("="*70)
    print("\nℹ️  This version loads pre-downloaded data from CSV files.")
    print("   No API key required!\n")
    
    print("✓ Initializing data loader...")
    
    loader = LocalDataLoader(raw_data_folder='data/raw')
    
    print("\n✓ Loading data from local CSV files...")
    df_raw = loader.load_all_countries()
    
    print(f"\n  Loaded {len(df_raw)} rows")
    print(f"  Countries: {sorted(df_raw['country'].unique())}")
    
    print("\n✓ Cleaning and aligning data...")
    df_clean = loader.clean_and_filter_data(df_raw)
    
    print("\n✓ Saving processed data...")
    loader.save_data(df_raw, 'raw_data.csv', subfolder='raw')
    loader.save_data(df_clean, 'processed_data.csv', subfolder='processed')
    
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

