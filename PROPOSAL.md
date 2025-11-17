**Monetary Policies and the Phillips Curve (Macroeconomics)**

**PROBLEM STATEMENT**

The Phillips Curve, the historical trade-off between inflation and unemployment, has shown significant variation across major developed economies since 2015, particularly during and after the COVID-19 pandemic. Central banks around the world adopted divergent policy frameworks and timings for rate adjustments between 2020 and 2025. Based on this relation, this project adopts a machine learning forecasting approach to investigate whether inflation can be predicted reliably one month ahead using unemployment, policy rates, and lagged indicators. I compare multiple ML models (Ridge/Lasso, Random Forest, XGBoost, LSTM) against a baseline econometric model (OLS), analyzing which one best captures the predictive dynamics and how policy frameworks affect forecasting performance across countries.

The motivation behind it relies on analyzing how recent historical events (and their consequent responses) have been able to modify one basic relationship “which ties at the heart of New Keynesian models and was widely thought to be flat before the pandemic” (Gudmundsson 2024).

**PLANNED APROACH**

**Data Collection:** Pull monthly time series (2015–2025) from the FRED API: unemployment rates, CPI/PCE inflation, and policy interest rates for the US, UK.

**Features:** Create lagged features, engineer policy regime indicators and define next-month inflation as target variable

**Machine Learning Models:** Train and compare multiple supervised regression models:
- Ridge & Lasso Regression: regularized linear models with cross-validated hyperparameters
- Random Forest Regressor: ensemble tree-based model capturing non-linear relationships
- XGBoost: gradient boosting for improved predictive power
- LSTM Neural Network (stretch goal): deep learning for sequential time series patterns

**Time Series Validation:** Implement time series cross-validation:
- Fold 1: Train on 2020-01 to 2020-12 (12months), test on 2021-01 to 2021-06 (6months)  
- Fold 2: Train expands to 2021-06, test in next 6 months, etc.  
- Across the full period, the minimum training window is 12months and each test window is 6months, continuing until the data ends.
- Prevent look-ahead bias: train only on past data, validate on future data

**Model Comparison & Evaluation:** Compare all models using:
- MAE
- RMSE
- Directional Accuracy

**Visualization:** 
- Static plots for report: forecast vs actual time series, model comparison bar charts, feature importance rankings, residual diagnostics
- Interactive Plotly charts for demo: model predictions with confidence bands, policy event annotations, cross-country comparisons


**TECHNOLOGIES:** pandas, numpy, fredapi, scikit-learn, xgboost, statsmodels, matplotlib, seaborn, plotly, pytest.

**CHALLENGES**

- **Time series leakage:** Enforce strict temporal ordering using expanding/rolling window CV.
- **Data Gaps:** Not every country presents their information the same way (or there could even be missing data). Interpolation methods could be used to mitigate this problem.
- **API Rate Limits:** Implement caching and retry logic to avoid interruptions.
- **Model overfitting:** Use regularization (Ridge/Lasso), early stopping (XGBoost), and cross-validation.


**SUCCESS CRITERIA**

- Automated FRED pipeline with no data leakage
- Three+ ML models trained with hyperparameter tuning
- Clear evidence of which models outperform OLS
- Feature importance analysis showing which variables drive predictions
- Unit tests for data and model functions
- Analysis of model performance variation by country/policy framework



**STRETCH GOALS**
- Policy Event Analysis: Event-study windows around major rate decisions.
- Interactive Dashboard: Streamlit app with live FRED updates.
