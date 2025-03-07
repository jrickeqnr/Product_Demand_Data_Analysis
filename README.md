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
├── logs/                       # Log files directory
├── config.json                 # Configuration file
├── config_loader.py            # Configuration loader utilities
├── data_fetcher.py             # Data fetching utilities
├── fetch_data.py               # Script to fetch and save data
├── exploratory_analysis.py     # EDA script
├── model_builder.py            # Model building script
└── main.py                     # Main execution script
```

## Configuration System

The project uses a JSON configuration file (`config.json`) to control various aspects of the analysis pipeline. This allows you to customize parameters without modifying the code.

### Configuration Options

Major configuration sections include:

- **Data Options**: Date range, data frequency, missing value handling
- **Analysis Options**: Model types to run (absolute/percent change), feature selection, lag features, seasonality
- **Modeling Options**: Train/test split, models to use, hyperparameter optimization, ensemble settings
- **Target Fuels**: Enable/disable diesel and gasoline analysis
- **Visualization**: Chart types, formats, DPI settings
- **Logging**: Log level, file and console logging options
- **Paths**: Directories for data, charts, models, and evaluation outputs

### Creating/Modifying Configuration

To create a default configuration file:

```
python main.py --create-config
```

You can then edit the `config.json` file to customize your analysis.

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
3. Create a default configuration file:
   ```
   python main.py --create-config
   ```

### Running the Analysis Pipeline

To run the entire analysis pipeline:

```
python main.py
```

You can also specify a custom configuration file:

```
python main.py --config my_custom_config.json
```

To run specific parts of the pipeline:

```
# Skip data fetching (use existing data)
python main.py --skip-fetch

# Skip exploratory data analysis
python main.py --skip-eda

# Skip model building
python main.py --skip-model
```

### Individual Script Execution

You can also run individual scripts directly with a specific configuration:

```
python fetch_data.py --config my_config.json
python exploratory_analysis.py --config my_config.json
python model_builder.py --config my_config.json
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

## Configuration Examples

### Date Range Configuration

Limit analysis to a specific date range:

```json
{
  "data": {
    "date_range": {
      "start_date": "2018-01-01",
      "end_date": "2022-12-31"
    }
  }
}
```

### Model Selection

Choose which models to run:

```json
{
  "modeling": {
    "models": {
      "linear_regression": true,
      "elasticnet": true,
      "random_forest": true,
      "gradient_boosting": true,
      "xgboost": false
    }
  }
}
```

### Feature Engineering

Configure feature engineering options:

```json
{
  "analysis": {
    "lag_features": {
      "enabled": true,
      "max_lag": 6
    },
    "seasonality": {
      "enabled": true,
      "add_month_dummies": true,
      "add_quarter_dummies": false
    }
  }
}
```

## Next Steps

Potential enhancements to this project:

1. Incorporate additional data sources (e.g., vehicle fleet composition, demographic data)
2. Implement more sophisticated time series models (ARIMA, Prophet, LSTM)
3. Add scenario analysis capability to evaluate impact of economic shocks
4. Create interactive dashboard for visualization and forecasting