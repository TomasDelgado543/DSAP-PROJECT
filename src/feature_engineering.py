"""
Feature engineering module for Phillips Curve ML project.
Creates lagged features and prediction targets for supervised learning.
"""

import pandas as pd
import os


class FeatureEngineer:
    """
    Transforms raw time series data into supervised learning format.
    Creates lagged features and prediction targets.
    """
    
    def __init__(self, lags: list = [1, 3, 6, 12]):
        """
        Initialize feature engineer.
        
        Parameters:
        -----------
        lags : list
            List of lag periods to create (in months)
            Default: [1, 3, 6, 12] = 1-month, 3-month, 6-month, 12-month lags
        """
        self.lags = lags
    
    def create_lagged_features(self, df: pd.DataFrame, columns: list, lags: list = None) -> pd.DataFrame:
        """
        Create lagged features for specified columns.
        FIX: Properly handle index alignment per country.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe (must have 'country' column to handle groups)
        columns : list
            Column names to create lags for (e.g., ['unemployment', 'inflation'])
        lags : list, optional
            Override default lag periods
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with original + lagged features
        """
        if lags is None:
            lags = self.lags
        
        df = df.copy()
        
        # Create lags for each country separately (time series logic)
        for country in df['country'].unique():
            country_mask = df['country'] == country
            
            for col in columns:
                for lag in lags:
                    lag_col_name = f"{col}_lag_{lag}"
                    
                    # Get the country's data
                    country_data = df.loc[country_mask, col]
                    
                    # Shift it
                    shifted_data = country_data.shift(lag)
                    
                    # Assign back using proper index
                    df.loc[country_mask, lag_col_name] = shifted_data.values
        
        return df

    
    def create_target_variable(self, df: pd.DataFrame, target_col: str = 'inflation', 
                              forecast_horizon: int = 1) -> pd.DataFrame:
        """
        Create target variable: future value to predict.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        target_col : str
            Column name to create future values for (default: 'inflation')
        forecast_horizon : int
            How many months ahead to predict (default: 1 = next month)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with target variable added
        """
        df = df.copy()
        target_name = f"{target_col}_t+{forecast_horizon}"
        
        # Create target for each country separately
        for country in df['country'].unique():
            country_mask = df['country'] == country
            df.loc[country_mask, target_name] = df.loc[country_mask, target_col].shift(-forecast_horizon)
        
        return df
    
    def create_policy_change_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features capturing policy changes and regimes.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe (must have 'policy_rate' column)
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with additional policy features
        """
        df = df.copy()
        
        for country in df['country'].unique():
            country_mask = df['country'] == country
            
            # Monthly change in policy rate (derivative)
            df.loc[country_mask, 'policy_rate_change'] = df.loc[country_mask, 'policy_rate'].diff()
            
            # Cumulative change over 12 months (annual policy stance)
            df.loc[country_mask, 'policy_rate_12m_change'] = df.loc[country_mask, 'policy_rate'].diff(12)
            
            # Binary indicators for tightening vs easing
            df.loc[country_mask, 'is_tightening'] = (df.loc[country_mask, 'policy_rate_change'] > 0).astype(int)
            df.loc[country_mask, 'is_easing'] = (df.loc[country_mask, 'policy_rate_change'] < 0).astype(int)
        
        return df
    
    def create_regime_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add COVID regime indicator.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe with date column
        
        Returns:
        --------
        pd.DataFrame
            DataFrame with regime features added
        """
        df = df.copy()
        df['post_covid'] = (df['date'] >= '2020-03-01').astype(int)
        df['years_since_covid'] = (df['date'] - pd.Timestamp('2020-03-01')).dt.days / 365.25
        df['years_since_covid'] = df['years_since_covid'].clip(lower=0)
        return df
    
    def engineer_features(self, df: pd.DataFrame, target_col: str = 'inflation',
                         forecast_horizon: int = 1) -> pd.DataFrame:
        """
        Complete feature engineering pipeline with detailed row tracking.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Raw processed data
        target_col : str
            Column to predict (default: 'inflation')
        forecast_horizon : int
            Months ahead to predict (default: 1)
        
        Returns:
        --------
        pd.DataFrame
            Feature-engineered dataset ready for ML
        """
        print("="*70)
        print("FEATURE ENGINEERING (WITH ROW TRACKING)")
        print("="*70)
        
        # Sort by country and date
        df = df.sort_values(['country', 'date']).reset_index(drop=True)
        
        # TRACKING: Start
        print("\n[TRACKING] Starting rows:")
        for country in sorted(df['country'].unique()):
            c_data = df[df['country'] == country]
            print(f"  {country}: {len(c_data)} rows, date range {c_data['date'].min().date()} to {c_data['date'].max().date()}")
        
        print("\n✓ Creating lagged features...")
        print(f"  Lag periods: {self.lags} months")
        
        # Create lags for unemployment, inflation, policy rate
        df = self.create_lagged_features(
            df, 
            columns=['unemployment', 'inflation', 'policy_rate'],
            lags=self.lags
        )
        
        print(f"  Created {len(self.lags) * 3} lagged features")
        
        # TRACKING: After lags
        print("\n[TRACKING] After lag creation (before dropna):")
        for country in sorted(df['country'].unique()):
            c_data = df[df['country'] == country]
            nan_count = c_data[['unemployment_lag_12', 'inflation_lag_12', 'policy_rate_lag_12']].isna().any(axis=1).sum()
            print(f"  {country}: {len(c_data)} rows, {nan_count} rows with NaN in 12-month lags")
        
        print("\n✓ Creating target variable...")
        print(f"  Target: {target_col}_t+{forecast_horizon} (predict {forecast_horizon} month(s) ahead)")
        
        df = self.create_target_variable(df, target_col=target_col, 
                                        forecast_horizon=forecast_horizon)
        
        # TRACKING: After target
        print("\n[TRACKING] After target creation (before dropna):")
        for country in sorted(df['country'].unique()):
            c_data = df[df['country'] == country]
            nan_count = c_data[f'{target_col}_t+{forecast_horizon}'].isna().sum()
            print(f"  {country}: {len(c_data)} rows, {nan_count} rows with NaN in target")
        
        print("\n✓ Creating policy change features...")
        df = self.create_policy_change_features(df)
        print("  - policy_rate_change (monthly change)")
        print("  - policy_rate_12m_change (annual change)")
        print("  - is_tightening (binary: rate increased)")
        print("  - is_easing (binary: rate decreased)")
        
        # TRACKING: After policy features
        print("\n[TRACKING] After policy features (before dropna):")
        for country in sorted(df['country'].unique()):
            c_data = df[df['country'] == country]
            nan_count = c_data[['policy_rate_change', 'policy_rate_12m_change']].isna().any(axis=1).sum()
            print(f"  {country}: {len(c_data)} rows, {nan_count} rows with NaN in policy features")
        
        print("\n✓ Creating regime indicator features...")
        df = self.create_regime_features(df)
        print("  - post_covid (binary: 1 if date >= 2020-03-01, 0 otherwise)")
        print("  - years_since_covid (continuous: years elapsed since COVID start)")
        
        # Remove rows with NaN values created by lags/target
        print("\n✓ Removing rows with missing values from lags/target...")
        rows_before = len(df)
        
        # TRACKING: Before dropna (detailed)
        print(f"\n[TRACKING] Complete rows per country (before dropna):")
        for country in sorted(df['country'].unique()):
            c_data = df[df['country'] == country]
            complete_rows = c_data.dropna()
            print(f"  {country}: {len(c_data)} total → {len(complete_rows)} complete ({len(c_data) - len(complete_rows)} will be dropped)")
        
        df = df.dropna()
        rows_after = len(df)
        rows_removed = rows_before - rows_after
        
        print(f"  Rows before dropna: {rows_before}")
        print(f"  Rows after dropna: {rows_after}")
        print(f"  Rows removed: {rows_removed}")
        
        # TRACKING: After dropna (final)
        print(f"\n[TRACKING] FINAL rows per country:")
        for country in sorted(df['country'].unique()):
            c_data = df[df['country'] == country]
            print(f"  {country}: {len(c_data)} rows, date range {c_data['date'].min().date()} to {c_data['date'].max().date()}")
        
        # Summary statistics
        print("\n" + "-"*70)
        print("FINAL FEATURE SET")
        print("-"*70)
        
        # Get feature columns (everything except date, country, original target)
        original_cols = ['date', 'country', 'unemployment', 'inflation', 'policy_rate']
        feature_cols = [col for col in df.columns if col not in original_cols 
                       and col != f'{target_col}_t+{forecast_horizon}']
        
        print(f"\nFeatures ({len(feature_cols)} total):")
        for i, col in enumerate(feature_cols, 1):
            print(f"  {i:2d}. {col}")
        
        print(f"\nTarget: {target_col}_t+{forecast_horizon}")
        
        print(f"\nDataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"Countries: {sorted(df['country'].unique())}")
        
        print("="*70 + "\n")
        
        return df
    
    def get_feature_columns(self, df: pd.DataFrame, target_col: str = 'inflation',
                           forecast_horizon: int = 1) -> tuple:
        """
        Get lists of feature and target columns.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature-engineered dataframe
        target_col : str
            Name of target column
        forecast_horizon : int
            Forecast horizon
        
        Returns:
        --------
        tuple
            (feature_cols, target_col_name)
        """
        original_cols = ['date', 'country', 'unemployment', 'inflation', 'policy_rate']
        target_col_name = f'{target_col}_t+{forecast_horizon}'
        
        feature_cols = [col for col in df.columns if col not in original_cols 
                       and col != target_col_name]
        
        return feature_cols, target_col_name


def main():
    """
    Main execution: load processed data and engineer features.
    """
    print("\n" + "="*70)
    print("PHILLIPS CURVE ML PROJECT - FEATURE ENGINEERING")
    print("="*70 + "\n")
    
    # Load processed data
    input_path = os.path.join('data', 'processed', 'processed_data.csv')
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Processed data not found at {input_path}\n"
            "Run data_collection.py first."
        )
    
    print(f"✓ Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"  Shape: {df.shape}")
    print(f"  Countries: {sorted(df['country'].unique())}")
    
    # Initialize feature engineer
    engineer = FeatureEngineer(lags=[1, 3, 6, 12])
    
    # Run feature engineering
    df_engineered = engineer.engineer_features(df, target_col='inflation', forecast_horizon=1)
    
    # Save engineered features
    output_path = os.path.join('data', 'processed', 'engineered_features.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_engineered.to_csv(output_path, index=False)
    print(f"✓ Engineered features saved to: {output_path}")
    
    # Get feature and target column names
    feature_cols, target_col = engineer.get_feature_columns(df_engineered)
    
    print(f"\n✓ Ready for ML modeling!")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(df_engineered)}")
    
    print("\n" + "="*70)
    print("Next step: Run models.py to train ML models")
    print("="*70 + "\n")
    
    return df_engineered


if __name__ == "__main__":
    df = main()
