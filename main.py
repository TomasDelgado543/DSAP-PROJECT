"""
Phillips Curve ML Project - Main Entry Point

Runs complete pipeline:
1. Data collection from FRED
2. Feature engineering
3. Model training and evaluation
4. Visualization generation
"""

import os
import sys
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Import modules
from src import data_collection, feature_engineering, models, visualization

def main():
    """Execute complete ML pipeline."""
    print("\n" + "="*70)
    print("PHILLIPS CURVE ML PROJECT - FULL PIPELINE")
    print("="*70 + "\n")
    
    # Step 1: Data Collection
    print("STEP 1: Data Collection from FRED")
    print("-" * 70)
    df_raw = data_collection.main()
    
    # Step 2: Feature Engineering
    print("\nSTEP 2: Feature Engineering")
    print("-" * 70)
    df_features = feature_engineering.main()
    
    # Step 3: Model Training & Evaluation
    print("\nSTEP 3: Model Training & Country Comparison")
    print("-" * 70)
    models.main()
    
    # Step 4: Visualization
    print("\nSTEP 4: Generating Visualizations")
    print("-" * 70)
    visualization.main()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE!")
    print("="*70)
    print("\nResults saved to:")
    print("  - data/results/country_comparison_results.csv")
    print("  - data/results/visualizations/*.png")
    print("\n")

if __name__ == "__main__":
    
    main()
