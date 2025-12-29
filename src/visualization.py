"""
Visualization module for Phillips Curve ML project - Country Comparison.
Creates charts comparing model performance across UK and US.
"""

import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10


class CountryComparisonVisualizer:
    """
    Create visualizations comparing ML model performance across countries.
    """
    
    def __init__(self, results_dir: str = 'results', 
                 output_dir: str = 'results/visualizations'):
        """
        Initialize visualizer.
        
        Parameters:
        -----------
        results_dir : str
            Directory with model results
        output_dir : str
            Directory to save visualizations
        """
        self.results_dir = results_dir
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Load country comparison results
        self.results = pd.read_csv(os.path.join(results_dir, 'country_comparison_results.csv'))
        
        # Color scheme
        self.colors = {
            'UK': "#287cb8",  # UK Blue
            'US': "#ff0101"   # US Red
        }
    
    def plot_model_comparison_by_country(self) -> str:
        """
        Create 2x2 subplot comparing model performance metrics by country.
        
        Returns:
        --------
        str
            Path to saved figure
        """
        print("  Creating country-specific model comparison...")
        
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Model Performance Comparison: UK vs US (2016-2025)', 
                    fontsize=16, fontweight='bold')
        
        models = ['OLS', 'Ridge', 'Lasso', 'Random Forest', 'XGBoost']
        x_pos = np.arange(len(models))
        width = 0.35
        
        # MAE (Lower is better)
        ax = axes[0, 0]
        uk_mae = [self.results[(self.results['Country']=='UK') & 
                              (self.results['Model']==m)]['MAE'].values[0] for m in models]
        us_mae = [self.results[(self.results['Country']=='US') & 
                              (self.results['Model']==m)]['MAE'].values[0] for m in models]
        
        ax.bar(x_pos - width/2, uk_mae, width, label='UK', color=self.colors['UK'], alpha=0.8)
        ax.bar(x_pos + width/2, us_mae, width, label='US', color=self.colors['US'], alpha=0.8)
        ax.set_title('Mean Absolute Error (MAE) - Lower is Better', fontweight='bold', fontsize=12)
        ax.set_ylabel('MAE', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # R² (Higher is better)
        ax = axes[0, 1]
        uk_r2 = [self.results[(self.results['Country']=='UK') & 
                             (self.results['Model']==m)]['R²'].values[0] for m in models]
        us_r2 = [self.results[(self.results['Country']=='US') & 
                             (self.results['Model']==m)]['R²'].values[0] for m in models]
        
        ax.bar(x_pos - width/2, uk_r2, width, label='UK', color=self.colors['UK'], alpha=0.8)
        ax.bar(x_pos + width/2, us_r2, width, label='US', color=self.colors['US'], alpha=0.8)
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline (mean)')
        ax.set_title('R² Score - Higher is Better (0 = mean baseline)', fontweight='bold', fontsize=12)
        ax.set_ylabel('R²', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # RMSE (Lower is better)
        ax = axes[1, 0]
        uk_rmse = [self.results[(self.results['Country']=='UK') & 
                               (self.results['Model']==m)]['RMSE'].values[0] for m in models]
        us_rmse = [self.results[(self.results['Country']=='US') & 
                               (self.results['Model']==m)]['RMSE'].values[0] for m in models]
        
        ax.bar(x_pos - width/2, uk_rmse, width, label='UK', color=self.colors['UK'], alpha=0.8)
        ax.bar(x_pos + width/2, us_rmse, width, label='US', color=self.colors['US'], alpha=0.8)
        ax.set_title('Root Mean Squared Error (RMSE) - Lower is Better', fontweight='bold', fontsize=12)
        ax.set_ylabel('RMSE', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Directional Accuracy (Higher is better)
        ax = axes[1, 1]
        uk_dir = [self.results[(self.results['Country']=='UK') & 
                              (self.results['Model']==m)]['Dir.Accuracy (%)'].values[0] for m in models]
        us_dir = [self.results[(self.results['Country']=='US') & 
                              (self.results['Model']==m)]['Dir.Accuracy (%)'].values[0] for m in models]
        
        ax.bar(x_pos - width/2, uk_dir, width, label='UK', color=self.colors['UK'], alpha=0.8)
        ax.bar(x_pos + width/2, us_dir, width, label='US', color=self.colors['US'], alpha=0.8)
        ax.axhline(y=50, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Random (50%)')
        ax.set_title('Directional Accuracy - Higher is Better (50% = random)', fontweight='bold', fontsize=12)
        ax.set_ylabel('Directional Accuracy (%)', fontsize=11)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(models, rotation=45, ha='right')
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim([35, 70])
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, '01_country_model_comparison.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path}")
        plt.close()
        
        return output_path
    
    def plot_best_model_by_country(self) -> str:
        """
        Highlight best-performing model for each country.
        
        Returns:
        --------
        str
            Path to saved figure
        """
        print("  Creating best model comparison...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Best Model by Country', fontsize=16, fontweight='bold')
        
        # UK best models
        ax = axes[0]
        uk_data = self.results[self.results['Country'] == 'UK'].sort_values('MAE')
        colors_uk = [self.colors['UK'] if i == 0 else 'lightgray' for i in range(len(uk_data))]
        
        ax.barh(uk_data['Model'], uk_data['MAE'], color=colors_uk, alpha=0.8)
        ax.set_xlabel('MAE (Lower is Better)', fontsize=11, fontweight='bold')
        ax.set_title('UK: Best Model = ' + uk_data.iloc[0]['Model'], fontweight='bold', fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add R² annotations
        for i, (idx, row) in enumerate(uk_data.iterrows()):
            ax.text(row['MAE'] + 0.02, i, f"R²={row['R²']:.3f}", va='center', fontsize=9)
        
        # US best models
        ax = axes[1]
        us_data = self.results[self.results['Country'] == 'US'].sort_values('MAE')
        colors_us = [self.colors['US'] if i == 0 else 'lightgray' for i in range(len(us_data))]
        
        ax.barh(us_data['Model'], us_data['MAE'], color=colors_us, alpha=0.8)
        ax.set_xlabel('MAE (Lower is Better)', fontsize=11, fontweight='bold')
        ax.set_title('US: Best Model = ' + us_data.iloc[0]['Model'], fontweight='bold', fontsize=12)
        ax.invert_yaxis()
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add R² annotations
        for i, (idx, row) in enumerate(us_data.iterrows()):
            ax.text(row['MAE'] + 0.02, i, f"R²={row['R²']:.3f}", va='center', fontsize=9)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, '02_best_model_by_country.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path}")
        plt.close()
        
        return output_path
    
    def plot_inflation_regimes(self) -> str:
        """
        Plot pre-COVID vs post-COVID inflation characteristics.
        
        Returns:
        --------
        str
            Path to saved figure
        """
        print("  Creating inflation regime comparison...")
        
        # Data from your results
        regime_data = {
            'UK': {
                'Pre-COVID': {'mean': 1.62, 'std': 0.84},
                'Post-COVID': {'mean': 4.38, 'std': 2.88}
            },
            'US': {
                'Pre-COVID': {'mean': 1.61, 'std': 0.86},
                'Post-COVID': {'mean': 4.22, 'std': 2.48}
            }
        }
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        fig.suptitle('Inflation Dynamics: Pre-COVID (2015-2020) vs Post-COVID (2020-2025)', 
                    fontsize=14, fontweight='bold')
        
        periods = ['Pre-COVID', 'Post-COVID']
        x_pos = np.arange(len(periods))
        width = 0.35
        
        # Mean inflation
        ax = axes[0]
        uk_means = [regime_data['UK'][p]['mean'] for p in periods]
        us_means = [regime_data['US'][p]['mean'] for p in periods]
        
        ax.bar(x_pos - width/2, uk_means, width, label='UK', color=self.colors['UK'], alpha=0.8)
        ax.bar(x_pos + width/2, us_means, width, label='US', color=self.colors['US'], alpha=0.8)
        ax.set_title('Mean Inflation by Period', fontweight='bold', fontsize=12)
        ax.set_ylabel('Inflation (%)', fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(periods)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(uk_means):
            ax.text(i - width/2, v + 0.1, f'{v:.2f}%', ha='center', fontsize=9)
        for i, v in enumerate(us_means):
            ax.text(i + width/2, v + 0.1, f'{v:.2f}%', ha='center', fontsize=9)
        
        # Volatility (standard deviation)
        ax = axes[1]
        uk_stds = [regime_data['UK'][p]['std'] for p in periods]
        us_stds = [regime_data['US'][p]['std'] for p in periods]
        
        ax.bar(x_pos - width/2, uk_stds, width, label='UK', color=self.colors['UK'], alpha=0.8)
        ax.bar(x_pos + width/2, us_stds, width, label='US', color=self.colors['US'], alpha=0.8)
        ax.set_title('Inflation Volatility (Std Dev) by Period', fontweight='bold', fontsize=12)
        ax.set_ylabel('Standard Deviation', fontsize=11, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(periods)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for i, v in enumerate(uk_stds):
            ax.text(i - width/2, v + 0.1, f'{v:.2f}', ha='center', fontsize=9)
        for i, v in enumerate(us_stds):
            ax.text(i + width/2, v + 0.1, f'{v:.2f}', ha='center', fontsize=9)
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, '03_inflation_regimes.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path}")
        plt.close()
        
        return output_path
    
    def plot_predictability_comparison(self) -> str:
        """
        Create scatter plot showing predictability (R²) vs complexity (model type).
        
        Returns:
        --------
        str
            Path to saved figure
        """
        print("  Creating predictability comparison scatter plot...")
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Plot each country
        for country in ['UK', 'US']:
            country_data = self.results[self.results['Country'] == country]
            
            ax.scatter(country_data['Model'], country_data['R²'], 
                      s=200, alpha=0.7, label=country, color=self.colors[country])
            
            # Add MAE as text
            for idx, row in country_data.iterrows():
                ax.annotate(f"MAE={row['MAE']:.2f}", 
                          (row['Model'], row['R²']),
                          textcoords="offset points", xytext=(0,10),
                          ha='center', fontsize=8)
        
        ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Baseline')
        ax.set_xlabel('Model', fontsize=12, fontweight='bold')
        ax.set_ylabel('R² Score', fontsize=12, fontweight='bold')
        ax.set_title('Model Predictability by Country (R² vs Model Complexity)', 
                    fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        
        output_path = os.path.join(self.output_dir, '04_predictability_scatter.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"    Saved: {output_path}")
        plt.close()
        
        return output_path
    
    # def create_interactive_country_comparison(self) -> str:
        """
        Create interactive Plotly chart for country comparison.
        
        Returns:
        --------
        str
            Path to saved HTML file
        """
        print("  Creating interactive country comparison...")
        
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('MAE by Model', 'R² Score by Model', 
                          'Directional Accuracy', 'RMSE by Model'),
            specs=[[{'type': 'bar'}, {'type': 'bar'}],
                   [{'type': 'bar'}, {'type': 'bar'}]]
        )
        
        models = ['OLS', 'Ridge', 'Lasso', 'Random Forest', 'XGBoost']
        
        for country in ['UK', 'US']:
            country_data = self.results[self.results['Country'] == country]
            
            # MAE
            fig.add_trace(
                go.Bar(name=country, x=models, 
                      y=[country_data[country_data['Model']==m]['MAE'].values[0] for m in models],
                      marker_color=self.colors[country]),
                row=1, col=1
            )
            
            # R²
            fig.add_trace(
                go.Bar(name=country, x=models, 
                      y=[country_data[country_data['Model']==m]['R²'].values[0] for m in models],
                      marker_color=self.colors[country], showlegend=False),
                row=1, col=2
            )
            
            # Dir Acc
            fig.add_trace(
                go.Bar(name=country, x=models, 
                      y=[country_data[country_data['Model']==m]['Dir.Accuracy (%)'].values[0] for m in models],
                      marker_color=self.colors[country], showlegend=False),
                row=2, col=1
            )
            
            # RMSE
            fig.add_trace(
                go.Bar(name=country, x=models, 
                      y=[country_data[country_data['Model']==m]['RMSE'].values[0] for m in models],
                      marker_color=self.colors[country], showlegend=False),
                row=2, col=2
            )
        
        fig.update_xaxes(tickangle=45)
        fig.update_layout(
            title_text='Phillips Curve ML: Country Comparison (Interactive)',
            height=800,
            showlegend=True,
            hovermode='x unified',
            barmode='group'
        )
        
        output_path = os.path.join(self.output_dir, '05_interactive_country_comparison.html')
        fig.write_html(output_path)
        print(f"    Saved: {output_path}")
        
        return output_path
    
    def generate_all_visualizations(self) -> None:
        """Generate all country comparison visualizations."""
        print("\n" + "="*70)
        print("GENERATING COUNTRY COMPARISON VISUALIZATIONS")
        print("="*70 + "\n")
        
        self.plot_model_comparison_by_country()
        self.plot_best_model_by_country()
        self.plot_inflation_regimes()
        self.plot_predictability_comparison()
        # self.create_interactive_country_comparison()
        
        print("\n" + "="*70)
        print(f"All visualizations saved to: {self.output_dir}")
        print("="*70 + "\n")


def main():
    """Main execution."""
    print("\n" + "="*70)
    print("PHILLIPS CURVE ML - COUNTRY COMPARISON VISUALIZATIONS")
    print("="*70 + "\n")
    
    results_dir = os.path.join('results')
    
    if not os.path.exists(os.path.join(results_dir, 'country_comparison_results.csv')):
        raise FileNotFoundError(
            f"Country comparison results not found at {results_dir}\n"
            "Run models_country_comparison.py first."
        )
    
    # Create visualizations
    viz = CountryComparisonVisualizer(results_dir=results_dir)
    viz.generate_all_visualizations()
    
    print("✓ Visualization complete!")
    print("\nGenerated files:")
    print("  - 01_country_model_comparison.png (4-metric comparison)")
    print("  - 02_best_model_by_country.png (winner highlight)")
    print("  - 03_inflation_regimes.png (pre/post COVID)")
    print("  - 04_predictability_scatter.png (R² analysis)")
    # print("  - 05_interactive_country_comparison.html (interactive demo)")
    print("\n")


if __name__ == "__main__":
    main()
