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
import argparse
from datetime import datetime
from config_loader import load_config

def setup_logging(config):
    """Set up logging based on configuration"""
    log_level = getattr(logging, config['logging']['level'])
    handlers = []
    
    if config['logging']['file_logging']:
        handlers.append(logging.FileHandler("model_building.log"))
    
    if config['logging']['console_logging']:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

class FuelDemandForecaster:
    def __init__(self, data_dir, target_fuel, config):
        """
        Initialize forecaster for a specific fuel type
        
        Args:
            data_dir (str): Directory containing input data files
            target_fuel (str): Target fuel type ('Diesel' or 'Gasoline')
            config (dict): Configuration dictionary
        """
        self.data_dir = data_dir
        self.target_fuel = target_fuel
        self.config = config
        
        # Set up paths
        self.models_dir = config['paths']['models_dir']
        self.evaluation_dir = config['paths']['evaluation_dir']
        
        # Create directories
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.evaluation_dir, exist_ok=True)
        
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
        
        # Configure visualization settings
        self.dpi = config['visualization']['dpi']
        self.save_formats = config['visualization']['save_formats']
        
    def save_figure(self, filename_base):
        """Save figure in all configured formats"""
        for fmt in self.save_formats:
            full_path = os.path.join(self.evaluation_dir, f"{filename_base}.{fmt}")
            plt.savefig(full_path, dpi=self.dpi)
        plt.close()
        
    def prepare_data(self, use_pct_change=True, feature_selection=None, lag_features=True, max_lag=None, add_seasonality=None):
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
            
        # Apply feature selection if enabled in config
        if self.config['analysis']['feature_selection']['enabled']:
            if feature_selection is not None:
                # Use provided feature list
                X = X[feature_selection]
            else:
                # Use correlation-based selection from config
                method = self.config['analysis']['feature_selection']['method']
                n_features = self.config['analysis']['feature_selection']['top_features']
                
                if method == 'correlation':
                    # Calculate correlations with target
                    if use_pct_change:
                        corr = X.corrwith(y).abs().sort_values(ascending=False)
                    else:
                        corr = X.corrwith(y).abs().sort_values(ascending=False)
                    
                    # Select top features
                    top_features = corr.head(n_features).index.tolist()
                    X = X[top_features]
                    logger.info(f"Selected {len(top_features)} features based on correlation with {self.target_fuel}")
            
        # Fill missing values with configured strategy
        fill_strategy = self.config['data']['fill_missing_strategy']
        if fill_strategy == 'median':
            X = X.fillna(X.median())
        elif fill_strategy == 'mean':
            X = X.fillna(X.mean())
        elif fill_strategy == 'zero':
            X = X.fillna(0)
        else:
            X = X.fillna(X.median())  # Default to median
        
        # Add lagged features for target variable if configured
        if lag_features and self.config['analysis']['lag_features']['enabled']:
            # Get max lag from config or use default
            if max_lag is None:
                max_lag = self.config['analysis']['lag_features']['max_lag']
            
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
                
        # Add seasonality features if configured
        if add_seasonality is None:
            add_seasonality = self.config['analysis']['seasonality']['enabled']
            
        if add_seasonality:
            X['month'] = X.index.month
            
            # Add quarter dummies if configured
            if self.config['analysis']['seasonality']['add_quarter_dummies']:
                X['quarter'] = X.index.quarter
                for quarter in range(1, 5):
                    X[f'quarter_{quarter}'] = (X['quarter'] == quarter).astype(int)
                X = X.drop(['quarter'], axis=1)
                
            # Add month dummies if configured
            if self.config['analysis']['seasonality']['add_month_dummies']:
                for month in range(2, 13):  # Skip first month to avoid multicollinearity
                    X[f'month_{month}'] = (X['month'] == month).astype(int)
                
            X = X.drop(['month'], axis=1)
            
        # Align X and y data
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Drop any remaining NaN values
        valid_idx = ~X.isna().any(axis=1)
        X = X.loc[valid_idx]
        y = y.loc[valid_idx]
        
        return X, y
        
    def train_test_split(self, X, y, test_size=None):
        """
        Split data into training and testing sets
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            test_size (int): Number of months to use for testing
            
        Returns:
            tuple: X_train, X_test, y_train, y_test
        """
        # Use test size from config if not provided
        if test_size is None:
            test_size = self.config['modeling']['train_test_split']['test_size']
            
        # Split based on time
        train_end = X.index[-test_size - 1]
        test_start = X.index[-test_size]
        
        X_train = X[:train_end]
        X_test = X[test_start:]
        y_train = y[:train_end]
        y_test = y[test_start:]
        
        return X_train, X_test, y_train, y_test
        
    def build_models(self, X, y, test_size=None, save_models=True):
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
        # Use test size from config if not provided
        if test_size is None:
            test_size = self.config['modeling']['train_test_split']['test_size']
            
        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size)
        
        logger.info(f"Training models with {len(X_train)} samples, testing with {len(X_test)} samples")
        
        # Define models to evaluate based on config
        models = {}
        
        if self.config['modeling']['models']['linear_regression']:
            models['Linear Regression'] = LinearRegression()
            
        if self.config['modeling']['models']['elasticnet']:
            models['ElasticNet'] = ElasticNet(random_state=42)
            
        if self.config['modeling']['models']['random_forest']:
            models['Random Forest'] = RandomForestRegressor(random_state=42)
            
        if self.config['modeling']['models']['gradient_boosting']:
            models['Gradient Boosting'] = GradientBoostingRegressor(random_state=42)
            
        if self.config['modeling']['models']['xgboost']:
            models['XGBoost'] = xgb.XGBRegressor(random_state=42)
        
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
                with open(os.path.join(self.models_dir, f'{self.target_fuel.lower()}_{name.replace(" ", "_").lower()}.pkl'), 'wb') as f:
                    pickle.dump(model, f)
                    
            logger.info(f"{name} - Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}, Test R²: {test_r2:.4f}")
        
        # Plot results
        if 'predictions' in self.config['visualization']['charts_to_generate']:
            self.plot_model_comparisons(results, predictions)
        
        # Save feature importance for tree-based models
        if 'feature_importance' in self.config['visualization']['charts_to_generate']:
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
        metrics_df.to_csv(os.path.join(self.evaluation_dir, f'{self.target_fuel.lower()}_model_performance.csv'))
        
        # Plot RMSE comparison
        plt.figure(figsize=(12, 6))
        metrics_df[['Train RMSE', 'Test RMSE']].plot(kind='bar')
        plt.title(f'Model RMSE Comparison - {self.target_fuel}')
        plt.ylabel('RMSE')
        plt.grid(True, axis='y')
        plt.tight_layout()
        self.save_figure(f'{self.target_fuel.lower()}_rmse_comparison')
        
        # Plot R² comparison
        plt.figure(figsize=(12, 6))
        metrics_df[['Train R²', 'Test R²']].plot(kind='bar')
        plt.title(f'Model R² Comparison - {self.target_fuel}')
        plt.ylabel('R²')
        plt.grid(True, axis='y')
        plt.tight_layout()
        self.save_figure(f'{self.target_fuel.lower()}_r2_comparison')
        
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
        self.save_figure(f'{self.target_fuel.lower()}_best_model_predictions')
        
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
        self.save_figure(f'{self.target_fuel.lower()}_test_predictions_timeseries')
        
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
            self.save_figure(f'{self.target_fuel.lower()}_{model_name.lower().replace(" ", "_")}_feature_importance')
            
            # Save importance to CSV for reference
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            feature_importance_df.to_csv(
                os.path.join(self.evaluation_dir, f'{self.target_fuel.lower()}_{model_name.lower().replace(" ", "_")}_feature_importance.csv'),
                index=False
            )
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            
    def optimize_model(self, model_type, X, y, test_size=None, param_grid=None):
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
        # Skip if optimization is disabled in config
        if not self.config['modeling']['hyperparameter_optimization']['enabled']:
            logger.info(f"Hyperparameter optimization is disabled in config. Skipping optimization for {model_type}.")
            return None, None, None
            
        # Check if this model is in the list of models to optimize
        if model_type not in self.config['modeling']['hyperparameter_optimization']['models_to_optimize']:
            logger.info(f"Model {model_type} is not in the list of models to optimize. Skipping optimization.")
            return None, None, None
            
        # Use test size from config if not provided
        if test_size is None:
            test_size = self.config['modeling']['train_test_split']['test_size']
            
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
        cv_folds = self.config['modeling']['hyperparameter_optimization']['cv_folds']
        tscv = TimeSeriesSplit(n_splits=cv_folds)
        
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
        with open(os.path.join(self.models_dir, f'{self.target_fuel.lower()}_{model_type}_optimized.pkl'), 'wb') as f:
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
        self.save_figure(f'{self.target_fuel.lower()}_{model_type}_optimized_predictions')
        
        return best_model, best_params, grid_search.cv_results_
        
    def ensemble_prediction(self, X, y, test_size=None, models_to_use=None):
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
        # Skip if ensemble is disabled in config
        if not self.config['modeling']['ensemble']['enabled']:
            logger.info("Ensemble modeling is disabled in config. Skipping ensemble prediction.")
            return None
            
        # Use test size from config if not provided
        if test_size is None:
            test_size = self.config['modeling']['train_test_split']['test_size']
            
        # Use models from config if not provided
        if models_to_use is None:
            models_to_use = self.config['modeling']['ensemble']['models_to_include']
            
        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size)
        
        # Train individual models
        trained_models = {}
        predictions = {}
        
        for model_type in models_to_use:
            # Load optimized model if available
            model_path = os.path.join(self.models_dir, f'{self.target_fuel.lower()}_{model_type}_optimized.pkl')
            
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
        self.save_figure(f'{self.target_fuel.lower()}_ensemble_predictions')
        
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
        
        with open(os.path.join(self.models_dir, f'{self.target_fuel.lower()}_ensemble_results.pkl'), 'wb') as f:
            pickle.dump(ensemble_results, f)
            
        return ensemble_results

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Build forecasting models for fuel demand')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    global logger
    logger = setup_logging(config)
    
    try:
        logger.info("Starting model building process...")
        
        # Check which model types to run from config
        model_types_to_run = config['analysis']['models_to_run']
        
        # Process diesel if enabled
        if config['targets']['diesel']:
            logger.info("Building models for Diesel demand...")
            diesel_forecaster = FuelDemandForecaster(
                data_dir=config['paths']['data_dir'],
                target_fuel='Diesel',
                config=config
            )
            
            # Build models for absolute values if configured
            if 'absolute' in model_types_to_run:
                logger.info("Building absolute value models for Diesel...")
                X_diesel, y_diesel = diesel_forecaster.prepare_data(use_pct_change=False)
                
                # Build and evaluate models
                diesel_results, diesel_predictions = diesel_forecaster.build_models(X_diesel, y_diesel)
                
                # Find best model for optimization
                best_model_diesel = max(diesel_results.items(), key=lambda x: x[1]['test_r2'])[0]
                
                # Map model name to model type for optimization
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
                    
                # Optimize model if configured
                if config['modeling']['hyperparameter_optimization']['enabled']:
                    logger.info(f"Optimizing {model_type} model for Diesel absolute values...")
                    diesel_forecaster.optimize_model(model_type, X_diesel, y_diesel)
                
                # Build ensemble model if configured
                if config['modeling']['ensemble']['enabled']:
                    logger.info("Building ensemble model for Diesel absolute values...")
                    diesel_forecaster.ensemble_prediction(X_diesel, y_diesel)
            
            # Build models for percent change if configured
            if 'percent_change' in model_types_to_run:
                logger.info("Building percent change models for Diesel...")
                X_diesel_pct, y_diesel_pct = diesel_forecaster.prepare_data(use_pct_change=True)
                
                # Build and evaluate models
                diesel_pct_results, diesel_pct_predictions = diesel_forecaster.build_models(
                    X_diesel_pct, y_diesel_pct
                )
                
                # Find best model for optimization
                best_model_diesel_pct = max(diesel_pct_results.items(), key=lambda x: x[1]['test_r2'])[0]
                
                # Map model name to model type for optimization
                if best_model_diesel_pct == 'Linear Regression':
                    model_type = 'elasticnet'
                elif best_model_diesel_pct == 'Random Forest':
                    model_type = 'randomforest'
                elif best_model_diesel_pct == 'Gradient Boosting':
                    model_type = 'gbm'
                elif best_model_diesel_pct == 'XGBoost':
                    model_type = 'xgboost'
                else:
                    model_type = 'elasticnet'
                
                # Optimize model if configured
                if config['modeling']['hyperparameter_optimization']['enabled']:
                    logger.info(f"Optimizing {model_type} model for Diesel percent changes...")
                    diesel_forecaster.optimize_model(model_type, X_diesel_pct, y_diesel_pct)
                
                # Build ensemble model if configured
                if config['modeling']['ensemble']['enabled']:
                    logger.info("Building ensemble model for Diesel percent changes...")
                    diesel_forecaster.ensemble_prediction(X_diesel_pct, y_diesel_pct)
        
        # Process gasoline if enabled
        if config['targets']['gasoline']:
            logger.info("Building models for Gasoline demand...")
            gasoline_forecaster = FuelDemandForecaster(
                data_dir=config['paths']['data_dir'],
                target_fuel='Gasoline',
                config=config
            )
            
            # Build models for absolute values if configured
            if 'absolute' in model_types_to_run:
                logger.info("Building absolute value models for Gasoline...")
                X_gasoline, y_gasoline = gasoline_forecaster.prepare_data(use_pct_change=False)
                
                # Build and evaluate models
                gasoline_results, gasoline_predictions = gasoline_forecaster.build_models(X_gasoline, y_gasoline)
                
                # Find best model for optimization
                best_model_gasoline = max(gasoline_results.items(), key=lambda x: x[1]['test_r2'])[0]
                
                # Map model name to model type for optimization
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
                    
                # Optimize model if configured
                if config['modeling']['hyperparameter_optimization']['enabled']:
                    logger.info(f"Optimizing {model_type} model for Gasoline absolute values...")
                    gasoline_forecaster.optimize_model(model_type, X_gasoline, y_gasoline)
                
                # Build ensemble model if configured
                if config['modeling']['ensemble']['enabled']:
                    logger.info("Building ensemble model for Gasoline absolute values...")
                    gasoline_forecaster.ensemble_prediction(X_gasoline, y_gasoline)
            
            # Build models for percent change if configured
            if 'percent_change' in model_types_to_run:
                logger.info("Building percent change models for Gasoline...")
                X_gasoline_pct, y_gasoline_pct = gasoline_forecaster.prepare_data(use_pct_change=True)
                
                # Build and evaluate models
                gasoline_pct_results, gasoline_pct_predictions = gasoline_forecaster.build_models(
                    X_gasoline_pct, y_gasoline_pct
                )
                
                # Find best model for optimization
                best_model_gasoline_pct = max(gasoline_pct_results.items(), key=lambda x: x[1]['test_r2'])[0]
                
                # Map model name to model type for optimization
                if best_model_gasoline_pct == 'Linear Regression':
                    model_type = 'elasticnet'
                elif best_model_gasoline_pct == 'Random Forest':
                    model_type = 'randomforest'
                elif best_model_gasoline_pct == 'Gradient Boosting':
                    model_type = 'gbm'
                elif best_model_gasoline_pct == 'XGBoost':
                    model_type = 'xgboost'
                else:
                    model_type = 'elasticnet'
                
                # Optimize model if configured
                if config['modeling']['hyperparameter_optimization']['enabled']:
                    logger.info(f"Optimizing {model_type} model for Gasoline percent changes...")
                    gasoline_forecaster.optimize_model(model_type, X_gasoline_pct, y_gasoline_pct)
                
                # Build ensemble model if configured
                if config['modeling']['ensemble']['enabled']:
                    logger.info("Building ensemble model for Gasoline percent changes...")
                    gasoline_forecaster.ensemble_prediction(X_gasoline_pct, y_gasoline_pct)
        
        logger.info("Model building process completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in model building process: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()