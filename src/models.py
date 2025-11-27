"""
Country-specific model comparison with regime analysis and future predictions.
Trains separate models for US and UK to compare policy framework effects.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
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
        self.final_models = {}
    
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
    
    def diagnose_data_range(self, X, y, countries, dates):
        """Check data range - see if model sees recent data"""
        print("\n" + "="*70)
        print("DATA RANGE DIAGNOSTIC - CHECKING TRAINING DATA COVERAGE")
        print("="*70)
        
        for country in sorted(countries.unique()):
            country_mask = (countries == country)
            dates_country = dates[country_mask].reset_index(drop=True)
            
            print(f"\n{country}:")
            print(f"  Full dataset: {dates_country.min().date()} to {dates_country.max().date()}")
            print(f"  Total observations: {len(dates_country)}")
            
            # Time series CV split
            splitter = TimeSeriesSplitterByCountry(test_size=6, min_train_size=24)
            
            fold_count = 0
            for train_idx, test_idx in splitter.split(X[country_mask], y[country_mask], 
                                                      pd.Series([country] * len(X[country_mask]))):
                fold_count += 1
            
            # Show last fold
            fold = 0
            for train_idx, test_idx in splitter.split(X[country_mask], y[country_mask], 
                                                      pd.Series([country] * len(X[country_mask]))):
                fold += 1
                if fold == fold_count:  # Last fold
                    train_dates = dates_country.iloc[train_idx]
                    test_dates = dates_country.iloc[test_idx]
                    
                    print(f"\n  Last CV Fold ({fold_count}):")
                    print(f"    Train: {train_dates.min().date()} to {train_dates.max().date()}")
                    print(f"    Test:  {test_dates.min().date()} to {test_dates.max().date()}")
                    print(f"    ⚠️  Missing from CV training: {test_dates.max().date()} to {dates_country.max().date()}")
                    
                    missing_months = (dates_country.max() - test_dates.max()).days / 30.44
                    print(f"    → {missing_months:.1f} months of data not seen during training!")
        
        print(f"\n{'='*70}\n")
    
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
            
            # Filter out rows with NaN target
            valid_target_mask = ~y_country.isna()
            
            X_country = X_country[valid_target_mask].reset_index(drop=True)
            y_country = y_country[valid_target_mask].reset_index(drop=True)
            dates_country = dates_country[valid_target_mask].reset_index(drop=True)
            
            # Drop rows with NaN features
            valid_features_mask = ~X_country.isna().any(axis=1)
            
            X_country = X_country[valid_features_mask].reset_index(drop=True)
            y_country = y_country[valid_features_mask].reset_index(drop=True)
            dates_country = dates_country[valid_features_mask].reset_index(drop=True)
            
            print(f"\n  Total observations (after feature cleaning): {len(X_country)}")
            print(f"  Observations with valid target: {len(X_country)}")
            print(f"  Date range: {dates_country.min().date()} to {dates_country.max().date()}")
                        
            # Sort by date
            sort_idx = dates_country.argsort()
            X_country = X_country.iloc[sort_idx].reset_index(drop=True)
            y_country = y_country.iloc[sort_idx].reset_index(drop=True)
            dates_country = dates_country.iloc[sort_idx].reset_index(drop=True)
            
            # Time series CV (use _train data only)
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
                    
                    all_results_by_model[model_name]['y_true'].extend(y_test.values)
                    all_results_by_model[model_name]['y_pred'].extend(y_pred)
                
                fold += 1
            
            # Aggregate results for this country
            print(f"\n  {country} - CROSS-VALIDATION RESULTS:")
            print(f"  {'-'*70}")
            
            best_mae = float('inf')
            best_model_name = None
            
            for model_name in ['OLS', 'Ridge', 'Lasso', 'Random Forest', 'XGBoost']:
                y_true_array = np.array(all_results_by_model[model_name]['y_true'])
                y_pred_array = np.array(all_results_by_model[model_name]['y_pred'])
                
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
                
                # Track best model (based on mae)
                if metrics['mae'] < best_mae:
                    best_mae = metrics['mae']
                    best_model_name = model_name
            
            self.results_by_country[country] = {
            'best_model': best_model_name,
            'X': X_country,  # ← Full data (includes March 2025)
            'y': y_country,  # ← Full data (March has NaN)
            'dates': dates_country
            }
        
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
    

    def train_final_models(self, X, y, countries, dates, feature_cols):
        """Train final models on COMPLETE dataset for each country"""
        print("\n" + "="*70)
        print("TRAINING FINAL MODELS ON COMPLETE DATASET")
        print("="*70)
        
        for country in sorted(countries.unique()):
            country_mask = (countries == country)
            X_country = X[country_mask].reset_index(drop=True)
            y_country = y[country_mask].reset_index(drop=True)
            dates_country = dates[country_mask].reset_index(drop=True)
            
            # Filter valid targets
            valid_target_mask = ~y_country.isna()
            X_country = X_country[valid_target_mask].reset_index(drop=True)
            y_country = y_country[valid_target_mask].reset_index(drop=True)
            dates_country = dates_country[valid_target_mask].reset_index(drop=True)
            
            # ⚠️ ADD THIS: Also filter valid features
            valid_features_mask = ~X_country.isna().any(axis=1)
            X_train_full = X_country[valid_features_mask]
            y_train_full = y_country[valid_features_mask]
            dates_train_full = dates_country[valid_features_mask]
            
            best_model_name = self.results_by_country[country]['best_model']
            
            # Get the model class
            if best_model_name == 'OLS':
                final_model = LinearRegression()
            elif best_model_name == 'Ridge':
                final_model = Ridge(alpha=1.0)
            elif best_model_name == 'Lasso':
                final_model = Lasso(alpha=0.01, max_iter=10000)
            elif best_model_name == 'Random Forest':
                final_model = RandomForestRegressor(n_estimators=100, max_depth=10, 
                                                random_state=self.random_state, n_jobs=-1)
            elif best_model_name == 'XGBoost':
                final_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, 
                                            random_state=self.random_state, verbosity=0)
            
            # Train on valid data
            final_model.fit(X_train_full, y_train_full)
            
            print(f"\n✓ {country} - Final {best_model_name} trained on {len(X_train_full)} observations")
            print(f"  Date range: {dates_train_full.min().date()} to {dates_train_full.max().date()}")
            
            # Store for predictions
            self.final_models[country] = {
                'model': final_model,
                'X_latest': X_train_full.iloc[-1],
                'date_latest': dates_train_full.iloc[-1],
                'inflation_latest': y_train_full.iloc[-1]
            }
        
        print()


    
    def predict_next_period(self):
        """Make predictions for next month (April 2025)"""
        print("\n" + "="*70)
        print("FUTURE INFLATION PREDICTIONS - APRIL 2025")
        print("="*70)
        
        predictions = {}
        
        for country in sorted(self.final_models.keys()):
            model_data = self.final_models[country]
            model = model_data['model']
            X_latest = model_data['X_latest'].values.reshape(1, -1)
            date_latest = model_data['date_latest']
            inflation_latest = model_data['inflation_latest']
            
            # Make prediction
            prediction = model.predict(X_latest)[0]
            
            # Next month (add ~30 days)
            next_date = pd.Timestamp(date_latest) + timedelta(days=30)
            
            print(f"\n{country}:")
            print(f"  Latest data available: {date_latest.date()}")
            print(f"  Last observed inflation: {inflation_latest:.2f}%")
            print(f"  Predicted inflation ({next_date.strftime('%B %Y')}): {prediction:.2f}%")
            print(f"  Change: {prediction - inflation_latest:+.2f} percentage points")
            
            predictions[country] = {
                'date': next_date,
                'prediction': prediction,
                'latest_inflation': inflation_latest,
                'change': prediction - inflation_latest
            }
        
        print(f"\n{'='*70}\n")
        
        return predictions
    
    def predict_april_2025(self, df_predict):
        """
        Use March 2025 features to predict April 2025 inflation
        """
        print("\n" + "="*70)
        print("APRIL 2025 INFLATION PREDICTIONS")
        print("="*70)
        print("\nUsing March 2025 features with trained models...\n")
        
        predictions = {}
        
        for country in sorted(df_predict['country'].unique()):
            country_data = df_predict[df_predict['country'] == country].reset_index(drop=True)
            
            if len(country_data) == 0:
                print(f"{country}: No prediction data available")
                continue
            
            # Get the model and data
            model = self.final_models[country]['model']
            date_march = country_data.iloc[-1]['date']
            
            # Get features (exclude non-feature columns)
            exclude_cols = ['date', 'country', 'unemployment', 'inflation', 'policy_rate', 'inflation_t+1']
            feature_cols = [col for col in country_data.columns if col not in exclude_cols]
            
            X_march = country_data[feature_cols].iloc[-1].values.reshape(1, -1)
            
            # Make prediction
            april_pred = model.predict(X_march)[0]
            march_inflation = country_data.iloc[-1]['inflation']
            
            date_april = pd.Timestamp(date_march) + timedelta(days=30)
            
            print(f"{country}:")
            print(f"  Data used: March 2025 ({date_march.date()})")
            print(f"  Current inflation (March): {march_inflation:.2f}%")
            print(f"  Predicted inflation (April 2025): {april_pred:.2f}%")
            print(f"  Expected change: {april_pred - march_inflation:+.2f} percentage points\n")
            
            predictions[country] = {
                'date': date_april,
                'prediction': april_pred,
                'march_inflation': march_inflation,
                'change': april_pred - march_inflation
            }
        
        print("="*70 + "\n")
        return predictions


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
    
    # Load TRAINING data (with targets)
    input_path_train = os.path.join('data', 'processed', 'engineered_features_train.csv')
    
    if not os.path.exists(input_path_train):
        raise FileNotFoundError(f"Training data not found at {input_path_train}")
    
    df_train = pd.read_csv(input_path_train)
    df_train['date'] = pd.to_datetime(df_train['date'])
    
    print(f"✓ Loaded TRAINING data: {len(df_train)} observations")
    print(f"  Date range: {df_train['date'].min().date()} to {df_train['date'].max().date()}")
    print(f"  Countries: {sorted(df_train['country'].unique())}")
    
    # Prepare data for CV
    comparison = CountryComparison()
    X, y, countries, dates, features = comparison.prepare_data(df_train)
    
    print(f"\n✓ Features: {len(features)}")
    
    # DIAGNOSTIC: Check data range coverage
    comparison.diagnose_data_range(X, y, countries, dates)
    
    # Run country-specific CV
    df_results = comparison.cross_validate_by_country(X, y, countries, dates)
    
    # Train final models on complete training dataset
    comparison.train_final_models(X, y, countries, dates, features)
    
    # Save CV results
    output_dir = os.path.join('data', 'results')
    os.makedirs(output_dir, exist_ok=True)
    df_results.to_csv(os.path.join(output_dir, 'country_comparison_results.csv'), index=False)
    
    # Analyze by period
    comparison.analyze_by_period(df_train)
    
    # LOAD PREDICTION DATA (March 2025, no targets)
    input_path_pred = os.path.join('data', 'processed', 'engineered_features_predict.csv')
    
    if os.path.exists(input_path_pred):
        df_predict = pd.read_csv(input_path_pred)
        df_predict['date'] = pd.to_datetime(df_predict['date'])
        
        print(f"\n✓ Loaded PREDICTION data: {len(df_predict)} observations")
        print(f"  Date range: {df_predict['date'].min().date()} to {df_predict['date'].max().date()}")
        
        # Make April 2025 predictions using March 2025 features
        predictions = comparison.predict_april_2025(df_predict)
    else:
        print("\n⚠️  Prediction data not found - skipping April 2025 forecast")
        predictions = None
    
    print("="*70)
    print("✓ Results saved to: country_comparison_results.csv")
    print("="*70 + "\n")
    
    return df_results, predictions


if __name__ == "__main__":
    results, predictions = main()
