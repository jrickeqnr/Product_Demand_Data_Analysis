#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Backtest Linear Models for Fuel Demand

This script backtests the linear models for diesel and gasoline demand
using coefficients from the model specification file and compares results
with actual values.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
from datetime import datetime

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette('Set2')

def load_data():
    """Load input features and demand data"""
    features_df = pd.read_csv('data/input_features_absolute.csv', parse_dates=['Date'])
    demand_df = pd.read_csv('data/fuel_demand_data.csv', parse_dates=['date'])
    
    # Rename columns to be consistent
    demand_df = demand_df.rename(columns={'date': 'Date'})
    
    # Set Date as index for both DataFrames
    features_df = features_df.set_index('Date')
    demand_df = demand_df.set_index('Date')
    
    print(f"Features data loaded: {features_df.shape[0]} rows, {features_df.shape[1]} columns")
    print(f"Demand data loaded: {demand_df.shape[0]} rows, {demand_df.shape[1]} columns")
    
    return features_df, demand_df

def load_model_coefficients(filename='old_model.csv'):
    """Load and parse the model coefficients from the text file"""
    model_df = pd.read_csv(filename)
    
    # Create dictionaries to store coefficients for each product
    models = {}
    
    # Group by Product to separate different models
    for product, product_df in model_df.groupby('Product'):
        if product in ['Gasoline', 'Diesel']:
            models[product] = {
                'intercept': product_df[product_df['Category'] == 'Intercept']['Value'].values[0],
                'coefficients': {}
            }
            
            # Extract coefficients for variables
            for _, row in product_df[product_df['Category'] != 'Intercept'].iterrows():
                variable = row['Variable']
                value = row['Value']
                scaling = row['Scaling'] if row['Scaling'] != 0 else 1
                lag = int(row['Lag']) if row['Lag'] != 0 else 0
                
                models[product]['coefficients'][variable] = {
                    'value': value,
                    'scaling': scaling,
                    'lag': lag
                }
    
    return models

def create_lagged_features(df, max_lag=12):
    """Create lagged versions of all features"""
    df_lagged = df.copy()
    
    for col in df.columns:
        for lag in range(1, max_lag + 1):
            df_lagged[f"{col}_lag{lag}"] = df[col].shift(lag)
            
    return df_lagged

def prepare_data_for_model(features_df, demand_df, models):
    """Prepare data for modeling by combining and aligning features and demand"""
    # Merge features and demand data
    combined_df = pd.merge(features_df, demand_df, left_index=True, right_index=True, how='inner')
    
    # Create lagged features
    max_lag = max([max([coef['lag'] for coef in model['coefficients'].values()]) 
                   for model in models.values()])
    
    if max_lag > 0:
        combined_df = create_lagged_features(combined_df, max_lag)
    
    # Drop rows with NaN values (due to lagging)
    combined_df = combined_df.dropna()
    
    return combined_df

def predict_with_linear_model(data, model_spec):
    """Make predictions using the linear model specifications"""
    intercept = model_spec['intercept']
    coefficients = model_spec['coefficients']
    
    # Start with intercept
    predictions = pd.Series(intercept, index=data.index)
    
    # Add contribution from each variable
    for variable, coef_info in coefficients.items():
        value = coef_info['value']
        scaling = coef_info['scaling']
        lag = coef_info['lag']
        
        # Check if it's a seasonality variable (numbered 1-12 for months)
        if variable.isdigit() and 1 <= int(variable) <= 12:
            # Create month indicator (1 if matches, 0 otherwise)
            month_indicator = data.index.month == int(variable)
            contribution = month_indicator * value
        else:
            # For regular variables
            if lag > 0:
                col_name = f"{variable}_lag{lag}"
            else:
                col_name = variable
                
            if col_name in data.columns:
                # Apply scaling and multiply by coefficient
                contribution = data[col_name] / scaling * value
            else:
                print(f"Warning: Variable {col_name} not found in data")
                contribution = 0
                
        predictions += contribution
    
    return predictions

def evaluate_model(actual, predicted, model_name):
    """Calculate evaluation metrics for the model"""
    mse = mean_squared_error(actual, predicted)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    
    print(f"\n--- {model_name} Model Evaluation ---")
    print(f"MSE: {mse:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"MAE: {mae:.2f}")
    print(f"R²: {r2:.4f}")
    
    return {
        'model': model_name,
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def plot_results(actual, predicted, model_name, output_dir='charts'):
    """Plot actual vs predicted values and residuals"""
    os.makedirs(output_dir, exist_ok=True)
    
    # Create a DataFrame for plotting
    results_df = pd.DataFrame({
        'Actual': actual,
        'Predicted': predicted,
        'Residual': actual - predicted
    })
    
    # 1. Actual vs Predicted Time Series
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['Actual'], label='Actual', linewidth=2)
    plt.plot(results_df.index, results_df['Predicted'], label='Predicted', linewidth=2, linestyle='--')
    
    # Format x-axis to show dates nicely
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gcf().autofmt_xdate()
    
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.xlabel('Date')
    plt.ylabel('Demand')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_timeseries.png", dpi=300)
    
    # 2. Scatter plot of Actual vs Predicted
    plt.figure(figsize=(8, 8))
    max_val = max(results_df['Actual'].max(), results_df['Predicted'].max()) * 1.05
    min_val = min(results_df['Actual'].min(), results_df['Predicted'].min()) * 0.95
    
    plt.scatter(results_df['Actual'], results_df['Predicted'], alpha=0.6)
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2)
    
    plt.title(f'{model_name} - Actual vs Predicted')
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_scatter.png", dpi=300)
    
    # 3. Residuals over time
    plt.figure(figsize=(12, 6))
    plt.plot(results_df.index, results_df['Residual'], label='Residual', color='darkred')
    plt.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    
    # Format x-axis
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(interval=6))
    plt.gcf().autofmt_xdate()
    
    plt.title(f'{model_name} - Residuals')
    plt.xlabel('Date')
    plt.ylabel('Residual (Actual - Predicted)')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_residuals.png", dpi=300)
    
    # 4. Residual distribution
    plt.figure(figsize=(10, 6))
    sns.histplot(results_df['Residual'], kde=True, color='darkblue')
    plt.axvline(x=0, color='r', linestyle='--')
    plt.title(f'{model_name} - Residual Distribution')
    plt.xlabel('Residual')
    plt.ylabel('Frequency')
    plt.tight_layout()
    plt.savefig(f"{output_dir}/{model_name}_residual_dist.png", dpi=300)
    
    return results_df

def main():
    """Main function to run the backtest"""
    print("Starting linear model backtest...")
    
    # Create output directories
    os.makedirs('charts', exist_ok=True)
    os.makedirs('evaluation', exist_ok=True)
    
    # Load data
    features_df, demand_df = load_data()
    
    # Load model coefficients
    models = load_model_coefficients()
    
    # Prepare data for modeling
    data = prepare_data_for_model(features_df, demand_df, models)
    
    # Store evaluation results and predictions
    all_results = []
    all_predictions = pd.DataFrame(index=data.index)
    
    # For each product model
    for product, model_spec in models.items():
        print(f"\nProcessing {product} model...")
        
        # Make predictions
        predictions = predict_with_linear_model(data, model_spec)
        all_predictions[f"{product}_Predicted"] = predictions
        
        # Add actuals
        all_predictions[f"{product}_Actual"] = data[product]
        
        # Evaluate model
        eval_results = evaluate_model(data[product], predictions, product)
        all_results.append(eval_results)
        
        # Plot results
        results_df = plot_results(data[product], predictions, product)
    
    # Save evaluation results
    eval_df = pd.DataFrame(all_results)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    eval_df.to_csv(f"evaluation/model_evaluation_{timestamp}.csv", index=False)
    print(f"Model evaluation saved to evaluation/model_evaluation_{timestamp}.csv")
    
    # Save predictions vs actuals
    all_predictions.to_csv(f"evaluation/predictions_vs_actuals_{timestamp}.csv")
    print(f"Predictions saved to evaluation/predictions_vs_actuals_{timestamp}.csv")
    
    print("\nBacktest completed successfully!")
    
if __name__ == "__main__":
    main()