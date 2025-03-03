# Fuel Demand Forecasting

This repository contains scripts for exploratory data analysis and forecasting of U.S. diesel and gasoline demand. The goal is to analyze factors that influence fuel demand and build predictive models for forecasting future demand patterns.

## Project Structure

```
├── data/                       # Data directory
│   ├── bloomberg_raw_data.csv  # Raw data from Bloomberg
│   ├── bloomberg_monthly_data.csv  # Aggregated monthly data
│   ├── fuel_demand_data.csv    # EIA fuel demand data
│   ├── input_features_absolute.csv  # Processed input features
│   ├── input_features_percent_change.csv  # Percent change features
│   ├── demand_absolute.csv     # Absolute demand values
│   ├── demand_percent_change.csv  # Demand percent changes
│   └── category_map.pkl        # Feature category mapping
├── charts/                     # EDA visualization outputs
├── models/                     # Trained forecasting models
├── evaluation/                 # Model evaluation outputs
├── data_fetcher.py             # Data fetching utilities
├── fetch_data.py               # Script to fetch and save data
├── exploratory_analysis.py     # EDA script
└── model_builder.py            # Model building script
```

## Data Sources

1. **Bloomberg Data:**
   - Economic indicators (GDP, retail sales, consumer sentiment, etc.)
   - Labor market metrics (unemployment rates, weekly wages, etc.)
   - Industrial metrics (production, raw steel, etc.)
   - Transport metrics (freight index, vehicle miles traveled, etc.)
   - Weather metrics (heating/cooling degree days)
   - Price metrics (gasoline and diesel retail prices)

2. **EIA Petroleum Data:**
   - U.S. Diesel Product Supplied (as a proxy for demand)
   - U.S. Gasoline Product Supplied (as a proxy for demand)

## Getting Started

### Prerequisites

- Python 3.8+
- Required packages: pandas, numpy, matplotlib, seaborn, scikit-learn, xgboost, scipy

### Installation

1. Clone this repository
2. Install required packages:
   ```
   pip install pandas numpy matplotlib seaborn scikit-learn xgboost scipy
   ```

### Data Fetching

To fetch and prepare the data:

```
python fetch_data.py
```

This script will:
- Connect to Bloomberg and EIA data sources
- Fetch and aggregate the data to monthly frequency
- Save the processed data to the `data` directory

### Exploratory Data Analysis

To perform exploratory data analysis:

```
python exploratory_analysis.py
```

This script will:
- Analyze data structure and patterns
- Plot demand trends and seasonality
- Analyze correlations between features and demand
- Perform PCA analysis to understand feature relationships
- Conduct rolling regression analysis
- Perform cross-correlation analysis to identify leading indicators

Visualizations will be saved to the `charts` directory.

### Model Building

To build and evaluate forecasting models:

```
python model_builder.py
```

This script will:
- Prepare data for model training (including feature engineering)
- Build and evaluate multiple forecasting models:
  - Linear Regression
  - ElasticNet
  - Random Forest
  - Gradient Boosting
  - XGBoost
- Optimize hyperparameters for the best performing model
- Create ensemble predictions
- Generate evaluation metrics and visualizations

Trained models will be saved to the `models` directory, and evaluation results will be saved to the `evaluation` directory.

## Analysis Approach

1. **Data Preparation:**
   - Aggregation to monthly frequency
   - Handling missing values
   - Feature engineering (lagged variables, seasonality features)
   - Calculating percent changes to capture growth rates

2. **Exploratory Analysis:**
   - Demand trend analysis
   - Seasonality patterns
   - Correlation analysis with potential predictors
   - Principal Component Analysis (PCA)
   - Rolling regression to identify changing relationships
   - Cross-correlation analysis to identify leading indicators

3. **Modeling Approaches:**
   - Linear models (Linear Regression, ElasticNet)
   - Tree-based models (Random Forest, Gradient Boosting, XGBoost)
   - Ensemble approach combining multiple models
   - Time series cross-validation for model evaluation
   - Both absolute value and percent change models

## Key Insights

The analysis and modeling in this repository can help answer:

1. What economic and market factors have the strongest influence on diesel and gasoline demand?
2. How do these relationships change over time?
3. Which models provide the most accurate forecasts of future demand?
4. What are the key leading indicators for fuel demand changes?
5. How do diesel and gasoline demand patterns differ in their drivers and seasonality?

## Next Steps

Potential enhancements to this project:

1. Incorporate additional data sources (e.g., vehicle fleet composition, demographic data)
2. Implement more sophisticated time series models (ARIMA, Prophet, LSTM)
3. Add scenario analysis capability to evaluate impact of economic shocks
4. Create interactive dashboard for visualization and forecasting