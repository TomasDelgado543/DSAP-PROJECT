"""
Machine Learning models for Phillips Curve inflation prediction.
Compares baseline OLS with Ridge, Lasso, Random Forest, and XGBoost.
"""

import pandas as pd
import numpy as np
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

# Statsmodels for OLS
from statsmodels.formula.api import ols


class TimeSeriesSplitter:
    """
    Custom time series cross-validator for proper temporal validation.
    Uses expanding window approach: training set grows, test set stays fixed size.
    """
    
    def __init__(self, test_size: int = 12, min_train_size: int = 24):
        """
        Initialize time series splitter.
        
        Parameters:
        -----------
        test_size : int
            Size of test set (in observations, default 12 = 1 year)
        min_train_size : int
            Minimum training set size (default 24 = 2 years)
        """
        self.test_size = test_size
        self.min_train_size = min_train_size
    
    def split(self, X: pd.DataFrame, y: pd.Series = None):
        """
        Generate train/test indices for time series cross-validation.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features dataframe
        y : pd.Series, optional
            Target (not used, for sklearn compatibility)
        
        Yields:
        -------
        tuple
            (train_indices, test_indices)
        """
        n_samples = len(X)
        n_splits = max(1, (n_samples - self.min_train_size - self.test_size) // self.test_size)
        
        for i in range(n_splits):
            train_end = self.min_train_size + (i * self.test_size)
            test_end = train_end + self.test_size
            
            train_idx = np.arange(0, train_end)
            test_idx = np.arange(train_end, test_end)
            
            yield train_idx, test_idx


class MLPipeline:
    """
    Complete ML pipeline for inflation prediction.
    Trains multiple models and compares performance.
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize ML pipeline.
        
        Parameters:
        -----------
        random_state : int
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.models = {}
        self.results = {}
        self.scalers = {}
        self.cv_results = {}
    
    def prepare_data(self, df: pd.DataFrame, target_col: str = 'inflation_t+1',
                    exclude_cols: list = None) -> tuple:
        """
        Prepare features and target from engineered features.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Feature-engineered dataframe
        target_col : str
            Name of target column
        exclude_cols : list
            Columns to exclude from features (date, country, etc.)
        
        Returns:
        --------
        tuple
            (X, y, feature_names, dates)
        """
        if exclude_cols is None:
            exclude_cols = ['date', 'country', 'unemployment', 'inflation', 'policy_rate']
        
        # Extract features and target
        feature_cols = [col for col in df.columns 
                       if col not in exclude_cols and col != target_col]
        
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        dates = df['date'].copy()
        countries = df['country'].copy()
        
        return X, y, feature_cols, dates, countries
    
    def train_ols_baseline(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Train OLS regression as baseline (econometric model).
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        
        Returns:
        --------
        dict
            Model and statistics
        """
        print("  Training OLS (baseline econometric model)...")
        
        model = LinearRegression()
        model.fit(X, y)
        
        # Predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        
        # Direction accuracy (did we get the sign right?)
        direction_correct = np.sum(np.sign(y - y.shift(1)) == np.sign(y_pred - y.shift(1)))
        direction_accuracy = direction_correct / (len(y) - 1) * 100
        
        result = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'y_pred': y_pred
        }
        
        print(f"    ✓ OLS - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Dir.Acc: {direction_accuracy:.1f}%")
        
        return result
    
    def train_ridge(self, X: pd.DataFrame, y: pd.Series, alpha: float = 1.0) -> dict:
        """Train Ridge Regression model."""
        print(f"  Training Ridge Regression (α={alpha})...")
        
        model = Ridge(alpha=alpha, random_state=self.random_state)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        direction_correct = np.sum(np.sign(y - y.shift(1)) == np.sign(y_pred - y.shift(1)))
        direction_accuracy = direction_correct / (len(y) - 1) * 100
        
        result = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'y_pred': y_pred
        }
        
        print(f"    ✓ Ridge - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Dir.Acc: {direction_accuracy:.1f}%")
        
        return result
    
    def train_lasso(self, X: pd.DataFrame, y: pd.Series, alpha: float = 0.01) -> dict:
        """Train Lasso Regression model."""
        print(f"  Training Lasso Regression (α={alpha})...")
        
        model = Lasso(alpha=alpha, random_state=self.random_state, max_iter=10000)
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        direction_correct = np.sum(np.sign(y - y.shift(1)) == np.sign(y_pred - y.shift(1)))
        direction_accuracy = direction_correct / (len(y) - 1) * 100
        
        result = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'y_pred': y_pred
        }
        
        print(f"    ✓ Lasso - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Dir.Acc: {direction_accuracy:.1f}%")
        
        return result
    
    def train_random_forest(self, X: pd.DataFrame, y: pd.Series, 
                           n_estimators: int = 100) -> dict:
        """Train Random Forest Regressor."""
        print(f"  Training Random Forest (n_estimators={n_estimators})...")
        
        model = RandomForestRegressor(
            n_estimators=n_estimators,
            random_state=self.random_state,
            n_jobs=-1,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2
        )
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        direction_correct = np.sum(np.sign(y - y.shift(1)) == np.sign(y_pred - y.shift(1)))
        direction_accuracy = direction_correct / (len(y) - 1) * 100
        
        result = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'y_pred': y_pred,
            'feature_importance': pd.Series(model.feature_importances_)
        }
        
        print(f"    ✓ RF - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Dir.Acc: {direction_accuracy:.1f}%")
        
        return result
    
    def train_xgboost(self, X: pd.DataFrame, y: pd.Series,
                     n_estimators: int = 100) -> dict:
        """Train XGBoost Regressor."""
        print(f"  Training XGBoost (n_estimators={n_estimators})...")
        
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=self.random_state,
            verbosity=0
        )
        model.fit(X, y)
        
        y_pred = model.predict(X)
        mae = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        r2 = r2_score(y, y_pred)
        direction_correct = np.sum(np.sign(y - y.shift(1)) == np.sign(y_pred - y.shift(1)))
        direction_accuracy = direction_correct / (len(y) - 1) * 100
        
        result = {
            'model': model,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'direction_accuracy': direction_accuracy,
            'y_pred': y_pred,
            'feature_importance': pd.Series(model.feature_importances_)
        }
        
        print(f"    ✓ XGB - MAE: {mae:.4f}, RMSE: {rmse:.4f}, R²: {r2:.4f}, Dir.Acc: {direction_accuracy:.1f}%")
        
        return result
    
    def train_all_models(self, X: pd.DataFrame, y: pd.Series) -> None:
        """
        Train all models.
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features
        y : pd.Series
            Target
        """
        print("\n" + "="*70)
        print("TRAINING ML MODELS")
        print("="*70 + "\n")
        
        # Train baseline
        self.results['OLS'] = self.train_ols_baseline(X, y)
        
        # Train regularized linear models
        self.results['Ridge'] = self.train_ridge(X, y, alpha=1.0)
        self.results['Lasso'] = self.train_lasso(X, y, alpha=0.01)
        
        # Train ensemble methods
        self.results['Random Forest'] = self.train_random_forest(X, y, n_estimators=100)
        self.results['XGBoost'] = self.train_xgboost(X, y, n_estimators=100)
    
    def compare_models(self) -> pd.DataFrame:
        """
        Compare all models and return results dataframe.
        
        Returns:
        --------
        pd.DataFrame
            Model comparison results
        """
        print("\n" + "="*70)
        print("MODEL COMPARISON")
        print("="*70)
        
        comparison = []
        
        for model_name, result in self.results.items():
            comparison.append({
                'Model': model_name,
                'MAE': result['mae'],
                'RMSE': result['rmse'],
                'R²': result['r2'],
                'Dir.Accuracy (%)': result['direction_accuracy']
            })
        
        df_comparison = pd.DataFrame(comparison)
        
        # Sort by MAE (lower is better)
        df_comparison = df_comparison.sort_values('MAE').reset_index(drop=True)
        
        print("\n")
        print(df_comparison.to_string(index=False))
        
        # Identify best model
        best_model = df_comparison.iloc[0]['Model']
        best_mae = df_comparison.iloc[0]['MAE']
        
        print(f"\n✓ Best Model: {best_model} (MAE: {best_mae:.4f})")
        print("="*70 + "\n")
        
        return df_comparison
    
    def get_feature_importance(self, feature_names: list) -> pd.DataFrame:
        """
        Extract feature importance from tree-based models.
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
        
        Returns:
        --------
        pd.DataFrame
            Feature importance dataframe
        """
        print("\n" + "="*70)
        print("FEATURE IMPORTANCE")
        print("="*70 + "\n")
        
        importance_data = {}
        
        # Random Forest
        if 'Random Forest' in self.results:
            rf_importance = self.results['Random Forest']['feature_importance']
            rf_importance.index = feature_names
            rf_importance = rf_importance.sort_values(ascending=False)
            importance_data['Random Forest'] = rf_importance
        
        # XGBoost
        if 'XGBoost' in self.results:
            xgb_importance = self.results['XGBoost']['feature_importance']
            xgb_importance.index = feature_names
            xgb_importance = xgb_importance.sort_values(ascending=False)
            importance_data['XGBoost'] = xgb_importance
        
        # Display top 10 features for each model
        for model_name, importance in importance_data.items():
            print(f"{model_name} - Top 10 Features:")
            for i, (feat, imp) in enumerate(importance.head(10).items(), 1):
                print(f"  {i:2d}. {feat:30s} {imp:.4f}")
            print()
        
        print("="*70 + "\n")
        
        return importance_data


def main():
    """
    Main execution: load data and train all models.
    """
    print("\n" + "="*70)
    print("PHILLIPS CURVE ML PROJECT - MODEL TRAINING")
    print("="*70 + "\n")
    
    # Load engineered features
    input_path = os.path.join('data', 'processed', 'engineered_features.csv')
    
    if not os.path.exists(input_path):
        raise FileNotFoundError(
            f"Engineered features not found at {input_path}\n"
            "Run feature_engineering.py first."
        )
    
    print(f"✓ Loading engineered features from: {input_path}")
    df = pd.read_csv(input_path)
    df['date'] = pd.to_datetime(df['date'])
    print(f"  Shape: {df.shape}")
    print(f"  Date range: {df['date'].min().date()} to {df['date'].max().date()}")
    
    # Initialize pipeline
    pipeline = MLPipeline(random_state=42)
    
    # Prepare data
    print("\n✓ Preparing features and target...")
    X, y, feature_names, dates, countries = pipeline.prepare_data(df)
    print(f"  Features: {len(feature_names)}")
    print(f"  Samples: {len(X)}")
    
    # Train all models (on full dataset for now - will add CV later)
    pipeline.train_all_models(X, y)
    
    # Compare models
    df_comparison = pipeline.compare_models()
    
    # Feature importance
    importance_dict = pipeline.get_feature_importance(feature_names)
    
    # Save results
    output_dir = os.path.join('data', 'results')
    os.makedirs(output_dir, exist_ok=True)
    
    print("✓ Saving results...")
    df_comparison.to_csv(os.path.join(output_dir, 'model_comparison.csv'), index=False)
    print(f"  Saved: {os.path.join(output_dir, 'model_comparison.csv')}")
    
    print("\n" + "="*70)
    print("MODEL TRAINING COMPLETE!")
    print("="*70)
    print(f"\nResults saved to: {output_dir}")
    print("Next step: Run visualization.py to create comparison charts")
    print("="*70 + "\n")
    
    return pipeline, df_comparison, importance_dict


if __name__ == "__main__":
    pipeline, comparison, importance = main()
