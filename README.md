# Phillips Curve ML Project

Comparing ML model performance for inflation forecasting: UK vs US (2015-2025)

## Setup

### Prerequisites
- Python 3.13.7
- Conda

### Installation

**Using Conda:**
conda env create -f environment.yml
conda activate phillips-ml-project

## Usage

**Run complete pipeline:**
python main.py


## Expected Output

Accuracy comparison between the models in each economy


## Project Structure

├── main.py # Entry point
├── environment.yml             # Conda dependencies
├── requirements.txt            # Pip dependencies
├── src/                        # Source code
│ ├── data_collection.py        # Data loading
│ ├── feature_engineering.py    # Feature Setting
│ ├── models.py                 # Model training
│ └── visualization.py          # Generating Visualizations
├── data/                       # Raw and processed data
├── results/                    # Outputs
└── notebooks                   # Exploration

## Results

**Key Findings:**
- UK: Linear models best (Lasso R²=0.94)
- US: Tree models best (XGBoost R²=0.75)
- Policy framework affects model performance

See `project_report.pdf` for full analysis.

## Author
Tomás Delgado
Université de Lausanne
Data Science Advance Programming
2025