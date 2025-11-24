"""
Country-specific model comparison with regime analysis.
Trains separate models for US and UK to compare policy framework effects.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb


class TimeSeriesSplitterByCountry:
    """Time series CV that keeps each country's data separate"""
    
    def __init__(self, test_size: int = 6, min_train_size: int = 24):
        self.test_size = test_size
        self.min_train_size = min_train_size
    
    def split(self, X, y, country_series):
        """Generate train/test splits per country"""
        for country in country_series.unique():
            country_mask = (country_series == country)
            country_indices = np.where(country_mask)[0]
            
            n_samples = len(country_indices)
            n_splits = max(1, (n_samples - self.min_train_size - self.test_size) // self.test_size)
            
            for i in range(n_splits):
                train_end = self.min_train_size + (i * self.test_size)
                test_end = train_end + self.test_size
                
                if test_end <= n_samples:
                    train_idx = country_indices[:train_end]
                    test_idx = country_indices[train_end:test_end]
                    
                    yield train_idx, test_idx


class CountryComparison:
    """Compare ML models across US and UK"""
    
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.results_by_country = {}
    
    def prepare_data(self, df, target_col='inflation_t+1'):
        """Prepare data for training"""
        exclude_cols = ['date', 'country', 'unemployment', 'inflation', 'policy_rate']
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and col != target_col]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        countries = df['country'].copy()
        dates = df['date'].copy()
        
        return X, y, countries, dates, feature_cols
    
    def evaluate_predictions(self, y_true, y_pred):
        """Calculate metrics"""
        if len(y_true) != len(y_pred):
            raise ValueError(f"y_true ({len(y_true)}) and y_pred ({len(y_pred)}) have different lengths!")
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        
        y_change_actual = np.diff(y_true)
        y_change_pred = np.diff(y_pred)
        direction_correct = np.sum(np.sign(y_change_actual) == np.sign(y_change_pred))
        direction_accuracy = direction_correct / len(y_change_actual) * 100 if len(y_change_actual) > 0 else 0
        
        return {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy
        }
    
    def train_models(self, X_train, y_train):
        """Train all models"""
        models = {
            'OLS': LinearRegression(),
            'Ridge': Ridge(alpha=1.0),
            'Lasso': Lasso(alpha=0.01, max_iter=10000),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=10, 
                                                   random_state=self.random_state, n_jobs=-1),
            'XGBoost': xgb.XGBRegressor(n_estimators=100, max_depth=5, 
                                       random_state=self.random_state, verbosity=0)
        }
        
        trained_models = {}
        for name, model in models.items():
            model.fit(X_train, y_train)
            trained_models[name] = model
        
        return trained_models
    
    def cross_validate_by_country(self, X, y, countries, dates):
        """Run CV separately for each country"""
        print("="*70)
        print("COUNTRY-SPECIFIC TIME SERIES CROSS-VALIDATION")
        print("="*70)
        
        results_list = []
        
        for country in sorted(countries.unique()):
            print(f"\n{'='*70}")
            print(f"COUNTRY: {country}")
            print(f"{'='*70}")
            
            country_mask = (countries == country)
            X_country = X[country_mask].reset_index(drop=True)
            y_country = y[country_mask].reset_index(drop=True)
            dates_country = dates[country_mask].reset_index(drop=True)
            
            # Sort by date
            sort_idx = dates_country.argsort()
            X_country = X_country.iloc[sort_idx].reset_index(drop=True)
            y_country = y_country.iloc[sort_idx].reset_index(drop=True)
            dates_country = dates_country.iloc[sort_idx].reset_index(drop=True)
            
            print(f"\n  Total observations: {len(X_country)}")
            print(f"  Date range: {dates_country.min().date()} to {dates_country.max().date()}")
            
            # Time series CV
            splitter = TimeSeriesSplitterByCountry(test_size=6, min_train_size=24)
            fold = 1
            
            # Initialize storage for ALL folds (per model)
            all_results_by_model = {
                'OLS': {'y_true': [], 'y_pred': []},
                'Ridge': {'y_true': [], 'y_pred': []},
                'Lasso': {'y_true': [], 'y_pred': []},
                'Random Forest': {'y_true': [], 'y_pred': []},
                'XGBoost': {'y_true': [], 'y_pred': []}
            }
            
            for train_idx, test_idx in splitter.split(X_country, y_country, 
                                                      pd.Series([country] * len(X_country))):
                print(f"\n  Fold {fold}:")
                print(f"    Train: {len(train_idx)} ({dates_country.iloc[train_idx[0]].date()} to {dates_country.iloc[train_idx[-1]].date()})")
                print(f"    Test:  {len(test_idx)} ({dates_country.iloc[test_idx[0]].date()} to {dates_country.iloc[test_idx[-1]].date()})")
                
                X_train, X_test = X_country.iloc[train_idx], X_country.iloc[test_idx]
                y_train, y_test = y_country.iloc[train_idx], y_country.iloc[test_idx]
                
                # Train models
                trained = self.train_models(X_train, y_train)
                
                # Evaluate and store predictions for each model
                for model_name, model in trained.items():
                    y_pred = model.predict(X_test)
                    
                    # IMPORTANT: Append for THIS fold only
                    all_results_by_model[model_name]['y_true'].extend(y_test.values)
                    all_results_by_model[model_name]['y_pred'].extend(y_pred)
                
                fold += 1
            
            # Aggregate results for this country
            print(f"\n  {country} - CROSS-VALIDATION RESULTS:")
            print(f"  {'-'*70}")
            
            for model_name in ['OLS', 'Ridge', 'Lasso', 'Random Forest', 'XGBoost']:
                y_true_array = np.array(all_results_by_model[model_name]['y_true'])
                y_pred_array = np.array(all_results_by_model[model_name]['y_pred'])
                
                # Verify they have same length
                if len(y_true_array) != len(y_pred_array):
                    print(f"    ⚠️  {model_name}: Length mismatch! ({len(y_true_array)} vs {len(y_pred_array)})")
                    continue
                
                metrics = self.evaluate_predictions(y_true_array, y_pred_array)
                
                results_list.append({
                    'Country': country,
                    'Model': model_name,
                    'MAE': metrics['mae'],
                    'RMSE': metrics['rmse'],
                    'R²': metrics['r2'],
                    'Dir.Accuracy (%)': metrics['direction_accuracy']
                })
                
                print(f"    {model_name:15s}: MAE={metrics['mae']:7.2f}, RMSE={metrics['rmse']:7.2f}, R²={metrics['r2']:7.4f}, Dir.Acc={metrics['direction_accuracy']:5.1f}%")
        
        df_results = pd.DataFrame(results_list)
        
        # Print country comparison
        print(f"\n{'='*70}")
        print("COUNTRY COMPARISON (BEST MODEL BY COUNTRY)")
        print(f"{'='*70}\n")
        
        for country in sorted(df_results['Country'].unique()):
            country_df = df_results[df_results['Country'] == country].sort_values('MAE')
            best = country_df.iloc[0]
            print(f"{country}: {best['Model']:15s} - MAE: {best['MAE']:7.2f}, R²: {best['R²']:7.4f}, Dir.Acc: {best['Dir.Accuracy (%)']:5.1f}%")
        
        print()
        
        return df_results
    
    def analyze_by_period(self, df, target_col='inflation_t+1'):
        """Analyze model performance by pre/post COVID"""
        print("\n" + "="*70)
        print("INFLATION CHARACTERISTICS BY PERIOD")
        print("="*70 + "\n")
        
        for country in sorted(df['country'].unique()):
            country_data = df[df['country'] == country].copy()
            country_data['date'] = pd.to_datetime(country_data['date'])
            
            pre_covid = country_data[country_data['date'] < '2020-03-01']
            post_covid = country_data[country_data['date'] >= '2020-03-01']
            
            print(f"\n{country}:")
            print(f"  Pre-COVID (2015-2020):  {len(pre_covid)} observations")
            print(f"  Post-COVID (2020-2025): {len(post_covid)} observations")
            
            if len(pre_covid) > 0:
                pre_inflation_mean = pre_covid['inflation'].mean()
                pre_inflation_std = pre_covid['inflation'].std()
                print(f"    Inflation: mean={pre_inflation_mean:.2f}%, std={pre_inflation_std:.2f}%")
            
            if len(post_covid) > 0:
                post_inflation_mean = post_covid['inflation'].mean()
                post_inflation_std = post_covid['inflation'].std()
                print(f"    Inflation: mean={post_inflation_mean:.2f}%, std={post_inflation_std:.2f}%")
        
        print()


def main():
    """Main execution"""
    print("\n" + "="*70)
    print("PHILLIPS CURVE ML - COUNTRY COMPARISON (2015-2025)")
    print("="*70 + "\n")
    
    # Load data
    input_path = os.path.join('data', 'processed', 'engineered_features.csv')
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Data not found at {input_path}")
    
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    
    print(f"✓ Loaded {len(df)} observations")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    print(f"  Countries: {sorted(df['country'].unique())}")
    
    # Prepare data
    comparison = CountryComparison()
    X, y, countries, dates, features = comparison.prepare_data(df)
    
    print(f"\n✓ Features: {len(features)}")
    
    # Run country-specific CV
    df_results = comparison.cross_validate_by_country(X, y, countries, dates)
    
    # Analyze by period
    comparison.analyze_by_period(df)
    
    # Save results
    output_dir = os.path.join('data', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    df_results.to_csv(os.path.join(output_dir, 'country_comparison_results.csv'), index=False)
    
    print("="*70)
    print("✓ Results saved to: country_comparison_results.csv")
    print("="*70 + "\n")
    
    return df_results


if __name__ == "__main__":
    results = main()
