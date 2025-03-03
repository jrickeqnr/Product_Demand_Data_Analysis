import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, LinearRegression
import xgboost as xgb
import pickle
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("model_building.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Create directories for models and evaluation
os.makedirs('models', exist_ok=True)
os.makedirs('evaluation', exist_ok=True)

class FuelDemandForecaster:
    def __init__(self, data_dir='data', target_fuel='Diesel'):
        """
        Initialize forecaster for a specific fuel type
        
        Args:
            data_dir (str): Directory containing input data files
            target_fuel (str): Target fuel type ('Diesel' or 'Gasoline')
        """
        self.data_dir = data_dir
        self.target_fuel = target_fuel
        
        # Load data
        self.input_absolute = pd.read_csv(f'{data_dir}/input_features_absolute.csv', parse_dates=['Date'], index_col='Date')
        self.input_pct = pd.read_csv(f'{data_dir}/input_features_percent_change.csv', parse_dates=['date'], index_col='date')
        self.demand_absolute = pd.read_csv(f'{data_dir}/demand_absolute.csv', parse_dates=['date'], index_col='date')
        self.demand_pct = pd.read_csv(f'{data_dir}/demand_percent_change.csv', parse_dates=['date'], index_col='date')
        
        # Load category mapping
        with open(f'{data_dir}/category_map.pkl', 'rb') as f:
            self.category_map = pickle.load(f)
            
        # Set target variable
        self.target_absolute = self.demand_absolute[target_fuel]
        self.target_pct = self.demand_pct[target_fuel]
        
        logger.info(f"Initialized forecaster for {target_fuel} with {len(self.input_absolute)} data points")
        
    def prepare_data(self, use_pct_change=True, feature_selection=None, lag_features=True, max_lag=3, add_seasonality=True):
        """
        Prepare data for model training
        
        Args:
            use_pct_change (bool): Use percent change data instead of absolute values
            feature_selection (list): List of features to use (if None, use all)
            lag_features (bool): Whether to add lagged features
            max_lag (int): Maximum lag to add
            add_seasonality (bool): Whether to add seasonality features
            
        Returns:
            tuple: X, y data ready for modeling
        """
        # Select input data based on type
        if use_pct_change:
            X = self.input_pct.copy()
            y = self.target_pct.copy()
        else:
            X = self.input_absolute.copy()
            y = self.target_absolute.copy()
            
        # Apply feature selection if specified
        if feature_selection is not None:
            X = X[feature_selection]
            
        # Fill missing values with median
        X = X.fillna(X.median())
        
        # Add lagged features for target variable
        if lag_features:
            # Add target variable first
            X_with_target = X.copy()
            if use_pct_change:
                X_with_target[self.target_fuel] = self.target_pct
            else:
                X_with_target[self.target_fuel] = self.target_absolute
                
            # Add lags
            for lag in range(1, max_lag + 1):
                lag_cols = X_with_target.shift(lag)
                lag_cols = lag_cols.rename(columns={col: f"{col}_lag{lag}" for col in lag_cols.columns})
                X = pd.concat([X, lag_cols], axis=1)
                
        # Add seasonality features
        if add_seasonality:
            X['month'] = X.index.month
            
            # Add seasonal dummies
            X['quarter'] = X.index.quarter
            for quarter in range(1, 5):
                X[f'quarter_{quarter}'] = (X['quarter'] == quarter).astype(int)
                
            # Add month dummies (except one to avoid multicollinearity)
            for month in range(2, 13):
                X[f'month_{month}'] = (X['month'] == month).astype(int)
                
            # Remove the original columns
            X = X.drop(['month', 'quarter'], axis=1)
            
        # Align X and y data
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Drop any remaining NaN values
        valid_idx = ~X.isna().any(axis=1)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        return X, y
        
    def train_test_split(self, X, y, test_size=12):
        """
        Split data into training and testing sets
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            test_size (int): Number of months to use for testing
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Split based on time
        train_end = X.index[-test_size - 1]
        test_start = X.index[-test_size]
        
        X_train = X[:train_end]
        X_test = X[test_start:]
        y_train = y[:train_end]
        y_test = y[test_start:]
        
        return X_train, X_test, y_train, y_test
        
    def build_models(self, X, y, test_size=12, save_models=True):
        """
        Build and evaluate multiple forecasting models
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            test_size (int): Number of months to use for testing
            save_models (bool): Whether to save trained models
            
        Returns:
            dict: Dictionary of model evaluation results
        """
        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size)
        
        logger.info(f"Training models with {len(X_train)} samples, testing with {len(X_test)} samples")
        
        # Define models to evaluate
        models = {
            'Linear Regression': LinearRegression(),
            'ElasticNet': ElasticNet(random_state=42),
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'XGBoost': xgb.XGBRegressor(random_state=42)
        }
        
        # Train and evaluate models
        results = {}
        predictions = {}
        
        for name, model in models.items():
            logger.info(f"Training {name} model...")
            
            # Train model
            model.fit(X_train, y_train)
            
            # Make predictions
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            
            # Calculate evaluation metrics
            train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
            test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
            
            train_mae = mean_absolute_error(y_train, y_pred_train)
            test_mae = mean_absolute_error(y_test, y_pred_test)
            
            train_r2 = r2_score(y_train, y_pred_train)
            test_r2 = r2_score(y_test, y_pred_test)
            
            # Store results
            results[name] = {
                'train_rmse': train_rmse,
                'test_rmse': test_rmse,
                'train_mae': train_mae,
                'test_mae': test_mae,
                'train_r2': train_r2,
                'test_r2': test_r2
            }
            
            predictions[name] = {
                'y_train': y_train,
                'y_pred_train': y_pred_train,
                'y_test': y_test,
                'y_pred_test': y_pred_test
            }
            
            # Save model if requested
            if save_models:
                with open(f'models/{self.target_fuel.lower()}_{name.replace(" ", "_").lower()}.pkl', 'wb') as f:
                    pickle.dump(model, f)
                    
            logger.info(f"{name} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
        
        # Plot results
        self.plot_model_comparisons(results, predictions)
        
        # Save feature importance for tree-based models
        for name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            if name in models:
                self.plot_feature_importance(models[name], X.columns, name)
                
        return results, predictions
        
    def plot_model_comparisons(self, results, predictions):
        """
        Plot comparison of model performance
        
        Args:
            results (dict): Dictionary of model evaluation results
            predictions (dict): Dictionary of model predictions
        """
        # Create DataFrame from results
        metrics_df = pd.DataFrame({
            model: {
                'Train RMSE': results[model]['train_rmse'],
                'Test RMSE': results[model]['test_rmse'],
                'Train MAE': results[model]['train_mae'],
                'Test MAE': results[model]['test_mae'],
                'Train R²': results[model]['train_r2'],
                'Test R²': results[model]['test_r2']
            }
            for model in results.keys()
        }).T
        
        # Save metrics to CSV
        metrics_df.to_csv(f'evaluation/{self.target_fuel.lower()}_model_performance.csv')
        
        # Plot RMSE comparison
        plt.figure(figsize=(12, 6))
        metrics_df[['Train RMSE', 'Test RMSE']].plot(kind='bar')
        plt.title(f'Model RMSE Comparison - {self.target_fuel}')
        plt.ylabel('RMSE')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f'evaluation/{self.target_fuel.lower()}_rmse_comparison.png')
        plt.close()
        
        # Plot R² comparison
        plt.figure(figsize=(12, 6))
        metrics_df[['Train R²', 'Test R²']].plot(kind='bar')
        plt.title(f'Model R² Comparison - {self.target_fuel}')
        plt.ylabel('R²')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(f'evaluation/{self.target_fuel.lower()}_r2_comparison.png')
        plt.close()
        
        # Plot actual vs predicted for best model
        best_model = metrics_df['Test R²'].idxmax()
        
        plt.figure(figsize=(15, 6))
        
        # Training data
        plt.subplot(1, 2, 1)
        plt.scatter(predictions[best_model]['y_train'], predictions[best_model]['y_pred_train'])
        plt.plot([min(predictions[best_model]['y_train']), max(predictions[best_model]['y_train'])], 
                [min(predictions[best_model]['y_train']), max(predictions[best_model]['y_train'])], 
                'k--')
        plt.title(f'Training Data: Actual vs Predicted - {best_model}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
        
        # Test data
        plt.subplot(1, 2, 2)
        plt.scatter(predictions[best_model]['y_test'], predictions[best_model]['y_pred_test'])
        plt.plot([min(predictions[best_model]['y_test']), max(predictions[best_model]['y_test'])], 
                [min(predictions[best_model]['y_test']), max(predictions[best_model]['y_test'])], 
                'k--')
        plt.title(f'Test Data: Actual vs Predicted - {best_model}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(f'evaluation/{self.target_fuel.lower()}_best_model_predictions.png')
        plt.close()
        
        # Plot time series of actual vs predicted for test data
        plt.figure(figsize=(12, 6))
        plt.plot(predictions[best_model]['y_test'].index, predictions[best_model]['y_test'], 'b-', label='Actual')
        plt.plot(predictions[best_model]['y_test'].index, predictions[best_model]['y_pred_test'], 'r--', label='Predicted')
        plt.title(f'{self.target_fuel} - Test Data Predictions ({best_model})')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'evaluation/{self.target_fuel.lower()}_test_predictions_timeseries.png')
        plt.close()
        
    def plot_feature_importance(self, model, feature_names, model_name):
        """
        Plot feature importance for tree-based models
        
        Args:
            model: Trained model
            feature_names: Names of features
            model_name: Name of the model
        """
        try:
            # Get feature importance
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif model_name == 'XGBoost':
                # For XGBoost we need a different approach
                importance = model.get_booster().get_score(importance_type='gain')
                # Convert to proper format
                importance = np.array([importance.get(f, 0) for f in feature_names])
            else:
                logger.warning(f"Could not extract feature importance for {model_name}")
                return
                
            # Sort features by importance
            indices = np.argsort(importance)[::-1]
            
            # Plot the top 20 features
            top_n = min(20, len(feature_names))
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importance - {model_name} - {self.target_fuel}')
            plt.bar(range(top_n), importance[indices][:top_n], align='center')
            plt.xticks(range(top_n), [feature_names[i] for i in indices][:top_n], rotation=90)
            plt.tight_layout()
            plt.savefig(f'evaluation/{self.target_fuel.lower()}_{model_name.lower().replace(" ", "_")}_feature_importance.png')
            plt.close()
            
            # Save importance to CSV for reference
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            feature_importance_df.to_csv(
                f'evaluation/{self.target_fuel.lower()}_{model_name.lower().replace(" ", "_")}_feature_importance.csv',
                index=False
            )
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            
    def optimize_model(self, model_type, X, y, test_size=12, param_grid=None):
        """
        Optimize hyperparameters for a specific model
        
        Args:
            model_type (str): Type of model to optimize
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            test_size (int): Number of months to use for testing
            param_grid (dict): Parameter grid for GridSearchCV
            
        Returns:
            tuple: Best model, best parameters, CV results
        """
        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size)
        
        # Define model and default parameter grid
        if model_type == 'elasticnet':
            model = ElasticNet(random_state=42)
            if param_grid is None:
                param_grid = {
                    'alpha': [0.001, 0.01, 0.1, 0.5, 1.0],
                    'l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
                }
        elif model_type == 'randomforest':
            model = RandomForestRegressor(random_state=42)
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20, 30],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                }
        elif model_type == 'gbm':
            model = GradientBoostingRegressor(random_state=42)
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0]
                }
        elif model_type == 'xgboost':
            model = xgb.XGBRegressor(random_state=42)
            if param_grid is None:
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 5, 7],
                    'subsample': [0.8, 1.0],
                    'colsample_bytree': [0.8, 1.0]
                }
        else:
            raise ValueError(f"Unsupported model type: {model_type}")
            
        # Set up time series cross-validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        # Set up grid search
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        # Fit grid search
        logger.info(f"Optimizing {model_type} model...")
        grid_search.fit(X_train, y_train)
        
        # Get best model and parameters
        best_model = grid_search.best_estimator_
        best_params = grid_search.best_params_
        
        logger.info(f"Best {model_type} parameters: {best_params}")
        
        # Evaluate on test set
        y_pred = best_model.predict(X_test)
        test_rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        test_r2 = r2_score(y_test, y_pred)
        
        logger.info(f"Optimized {model_type} - Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
        
        # Save best model
        with open(f'models/{self.target_fuel.lower()}_{model_type}_optimized.pkl', 'wb') as f:
            pickle.dump(best_model, f)
        
        # Plot feature importance if applicable
        if model_type in ['randomforest', 'gbm', 'xgboost']:
            self.plot_feature_importance(best_model, X.columns, f"{model_type}_optimized")
            
        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, 'b-', label='Actual')
        plt.plot(y_test.index, y_pred, 'r--', label='Predicted')
        plt.title(f'{self.target_fuel} - Optimized {model_type.capitalize()} Predictions')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'evaluation/{self.target_fuel.lower()}_{model_type}_optimized_predictions.png')
        plt.close()
        
        return best_model, best_params, grid_search.cv_results_
        
    def ensemble_prediction(self, X, y, test_size=12, models_to_use=None):
        """
        Create an ensemble prediction by averaging results from multiple models
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            test_size (int): Number of months to use for testing
            models_to_use (list): List of model types to include in ensemble
            
        Returns:
            dict: Dictionary with ensemble results
        """
        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size)
        
        # Default models if not specified
        if models_to_use is None:
            models_to_use = ['elasticnet', 'randomforest', 'gbm', 'xgboost']
            
        # Train individual models
        trained_models = {}
        predictions = {}
        
        for model_type in models_to_use:
            # Load optimized model if available
            model_path = f'models/{self.target_fuel.lower()}_{model_type}_optimized.pkl'
            
            if os.path.exists(model_path):
                logger.info(f"Loading optimized {model_type} model...")
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            else:
                # Create a default model
                logger.info(f"No optimized {model_type} model found, training default model...")
                if model_type == 'elasticnet':
                    model = ElasticNet(random_state=42)
                elif model_type == 'randomforest':
                    model = RandomForestRegressor(random_state=42)
                elif model_type == 'gbm':
                    model = GradientBoostingRegressor(random_state=42)
                elif model_type == 'xgboost':
                    model = xgb.XGBRegressor(random_state=42)
                
                # Train the model
                model.fit(X_train, y_train)
                
            # Store trained model
            trained_models[model_type] = model
            
            # Make predictions
            predictions[model_type] = model.predict(X_test)
            
        # Create ensemble prediction (simple average)
        ensemble_pred = np.zeros_like(predictions[models_to_use[0]])
        for model_type in models_to_use:
            ensemble_pred += predictions[model_type]
        ensemble_pred /= len(models_to_use)
        
        # Evaluate ensemble
        ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
        ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
        ensemble_r2 = r2_score(y_test, ensemble_pred)
        
        logger.info(f"Ensemble model - Test RMSE: {ensemble_rmse:.4f}, Test R²: {ensemble_r2:.4f}")
        
        # Plot ensemble results
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, 'b-', label='Actual')
        plt.plot(y_test.index, ensemble_pred, 'r--', label='Ensemble Predicted')
        
        # Add individual model predictions
        for model_type in models_to_use:
            plt.plot(y_test.index, predictions[model_type], '--', alpha=0.3, label=f'{model_type.capitalize()} Predicted')
            
        plt.title(f'{self.target_fuel} - Ensemble Model Predictions')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(f'evaluation/{self.target_fuel.lower()}_ensemble_predictions.png')
        plt.close()
        
        # Save ensemble results
        ensemble_results = {
            'models_used': models_to_use,
            'ensemble_rmse': ensemble_rmse,
            'ensemble_mae': ensemble_mae,
            'ensemble_r2': ensemble_r2,
            'y_test': y_test,
            'ensemble_pred': ensemble_pred,
            'individual_preds': predictions
        }
        
        with open(f'models/{self.target_fuel.lower()}_ensemble_results.pkl', 'wb') as f:
            pickle.dump(ensemble_results, f)
            
        return ensemble_results

def main():
    try:
        logger.info("Starting model building process...")
        
        # Build models for Diesel demand
        logger.info("Building models for Diesel demand...")
        diesel_forecaster = FuelDemandForecaster(target_fuel='Diesel')
        
        # Prepare data
        X_diesel, y_diesel = diesel_forecaster.prepare_data(
            use_pct_change=False,  # Start with absolute values
            lag_features=True,
            max_lag=3,
            add_seasonality=True
        )
        
        # Build and evaluate models
        diesel_results, diesel_predictions = diesel_forecaster.build_models(X_diesel, y_diesel, test_size=12)
        
        # Optimize the best performing model type
        best_model_diesel = max(diesel_results.items(), key=lambda x: x[1]['test_r2'])[0]
        
        if best_model_diesel == 'Linear Regression':
            model_type = 'elasticnet'  # Use ElasticNet instead for hyperparameter tuning
        elif best_model_diesel == 'Random Forest':
            model_type = 'randomforest'
        elif best_model_diesel == 'Gradient Boosting':
            model_type = 'gbm'
        elif best_model_diesel == 'XGBoost':
            model_type = 'xgboost'
        else:
            model_type = 'elasticnet'  # Default
            
        logger.info(f"Optimizing {model_type} model for Diesel demand...")
        diesel_forecaster.optimize_model(model_type, X_diesel, y_diesel)
        
        # Try percent change model as well
        logger.info("Building percent change models for Diesel demand...")
        X_diesel_pct, y_diesel_pct = diesel_forecaster.prepare_data(
            use_pct_change=True,
            lag_features=True,
            max_lag=3,
            add_seasonality=True
        )
        
        diesel_pct_results, _ = diesel_forecaster.build_models(
            X_diesel_pct, y_diesel_pct, test_size=12
        )
        
        # Build ensemble model
        logger.info("Building ensemble model for Diesel demand...")
        diesel_forecaster.ensemble_prediction(X_diesel, y_diesel)
        
        # Repeat for Gasoline demand
        logger.info("Building models for Gasoline demand...")
        gasoline_forecaster = FuelDemandForecaster(target_fuel='Gasoline')
        
        # Prepare data
        X_gasoline, y_gasoline = gasoline_forecaster.prepare_data(
            use_pct_change=False,
            lag_features=True,
            max_lag=3,
            add_seasonality=True
        )
        
        # Build and evaluate models
        gasoline_results, gasoline_predictions = gasoline_forecaster.build_models(X_gasoline, y_gasoline, test_size=12)
        
        # Optimize the best performing model type
        best_model_gasoline = max(gasoline_results.items(), key=lambda x: x[1]['test_r2'])[0]
        
        if best_model_gasoline == 'Linear Regression':
            model_type = 'elasticnet'
        elif best_model_gasoline == 'Random Forest':
            model_type = 'randomforest'
        elif best_model_gasoline == 'Gradient Boosting':
            model_type = 'gbm'
        elif best_model_gasoline == 'XGBoost':
            model_type = 'xgboost'
        else:
            model_type = 'elasticnet'
            
        logger.info(f"Optimizing {model_type} model for Gasoline demand...")
        gasoline_forecaster.optimize_model(model_type, X_gasoline, y_gasoline)
        
        # Try percent change model as well
        logger.info("Building percent change models for Gasoline demand...")
        X_gasoline_pct, y_gasoline_pct = gasoline_forecaster.prepare_data(
            use_pct_change=True,
            lag_features=True,
            max_lag=3,
            add_seasonality=True
        )
        
        gasoline_pct_results, _ = gasoline_forecaster.build_models(
            X_gasoline_pct, y_gasoline_pct, test_size=12
        )
        
        # Build ensemble model
        logger.info("Building ensemble model for Gasoline demand...")
        gasoline_forecaster.ensemble_prediction(X_gasoline, y_gasoline)
        
        logger.info("Model building process completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in model building process: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()