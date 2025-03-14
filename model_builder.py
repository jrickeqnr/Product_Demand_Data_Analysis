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

def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")
    return directory

# Create base directories
ensure_directory('models')
ensure_directory('evaluation')

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
        
        # Create model-specific subdirectories
        self.model_dir = ensure_directory(os.path.join('models', target_fuel.lower()))
        self.eval_dir = ensure_directory(os.path.join('evaluation', target_fuel.lower()))
        
        # Load data
        self.input_absolute = pd.read_csv(f'{data_dir}/input_features_absolute.csv', parse_dates=['date'], index_col='date')
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
        
    def build_models(self, X, y, test_size=12, save_models=True, model_type='absolute', n_splits=5):
        """
        Build and evaluate multiple forecasting models with cross-validation and scaling
        """
        model_subdir = ensure_directory(os.path.join(self.model_dir, model_type))
        eval_subdir = ensure_directory(os.path.join(self.eval_dir, model_type))
        
        tscv = TimeSeriesSplit(n_splits=n_splits)
        
        logger.info(f"Training {model_type} models for {self.target_fuel} with {len(X)} samples using {n_splits}-fold cross-validation")
        
        # Define models with Pipeline including StandardScaler
        models = {
            'Linear Regression': Pipeline([
                ('scaler', StandardScaler()),
                ('model', LinearRegression())
            ]),
            'ElasticNet': Pipeline([
                ('scaler', StandardScaler()),
                ('model', ElasticNet(random_state=42, max_iter=10000, tol=0.001))
            ]),
            'Random Forest': Pipeline([
                ('scaler', StandardScaler()),
                ('model', RandomForestRegressor(random_state=42))
            ]),
            'Gradient Boosting': Pipeline([
                ('scaler', StandardScaler()),
                ('model', GradientBoostingRegressor(random_state=42))
            ]),
            'XGBoost': Pipeline([
                ('scaler', StandardScaler()),
                ('model', xgb.XGBRegressor(random_state=42))
            ])
        }
        
        results = {}
        predictions = {}
        
        full_dates = pd.date_range(start='2015-06-01', end=X.index[-1], freq='MS')
        timeseries_df = pd.DataFrame(index=full_dates)
        timeseries_df['date'] = full_dates
        
        for name, pipeline in models.items():
            logger.info(f"Training {name} model with rolling window and cross-validation...")
            
            fold_train_rmse = []
            fold_test_rmse = []
            fold_train_mae = []
            fold_test_mae = []
            fold_train_r2 = []
            fold_test_r2 = []
            
            all_test_preds = pd.Series(index=X.index)
            all_test_actual = pd.Series(index=X.index)
            
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
                
                pipeline.fit(X_train, y_train)
                
                y_pred_train = pipeline.predict(X_train)
                y_pred_test = pipeline.predict(X_test)
                
                fold_train_rmse.append(np.sqrt(mean_squared_error(y_train, y_pred_train)))
                fold_test_rmse.append(np.sqrt(mean_squared_error(y_test, y_pred_test)))
                fold_train_mae.append(mean_absolute_error(y_train, y_pred_train))
                fold_test_mae.append(mean_absolute_error(y_test, y_pred_test))
                fold_train_r2.append(r2_score(y_train, y_pred_train))
                fold_test_r2.append(r2_score(y_test, y_pred_test))
                
                all_test_preds.iloc[test_idx] = y_pred_test
                all_test_actual.iloc[test_idx] = y_test
                
            results[name] = {
                'train_rmse_mean': np.mean(fold_train_rmse),
                'train_rmse_std': np.std(fold_train_rmse),
                'test_rmse_mean': np.mean(fold_test_rmse),
                'test_rmse_std': np.std(fold_test_rmse),
                'train_mae_mean': np.mean(fold_train_mae),
                'test_mae_mean': np.mean(fold_test_mae),
                'train_r2_mean': np.mean(fold_train_r2),
                'test_r2_mean': np.mean(fold_test_r2)
            }
            
            final_X_train, final_X_test, final_y_train, final_y_test = self.train_test_split(X, y, test_size)
            pipeline.fit(final_X_train, final_y_train)
            final_y_pred_train = pipeline.predict(final_X_train)
            final_y_pred_test = pipeline.predict(final_X_test)
            
            predictions[name] = {
                'y_train': final_y_train,
                'y_pred_train': final_y_pred_train,
                'y_test': final_y_test,
                'y_pred_test': final_y_pred_test
            }
            
            full_pred = pipeline.predict(X)
            pred_series = pd.Series(full_pred, index=X.index)
            aligned_pred = timeseries_df.index.map(
                lambda x: pred_series[x] if x in pred_series.index else np.nan
            )
            timeseries_df[f"{self.target_fuel}_{name.replace(' ', '_')}"] = aligned_pred
            
            if save_models:
                model_path = os.path.join(model_subdir, f"{name.replace(' ', '_').lower()}.pkl")
                with open(model_path, 'wb') as f:
                    pickle.dump(pipeline, f)
                    
            logger.info(f"{name} - Mean CV Test RMSE: {results[name]['test_rmse_mean']:.4f} (±{results[name]['test_rmse_std']:.4f})")
        
        base_eval_dir = ensure_directory('evaluation')
        output_file = os.path.join(base_eval_dir, f'{self.target_fuel.lower()}_predictions_{model_type}.csv')
        timeseries_df.to_csv(output_file, index=False)
        
        self.plot_model_comparisons(results, predictions, model_type)
        for name in ['Random Forest', 'Gradient Boosting', 'XGBoost']:
            if name in models:
                self.plot_feature_importance(models[name].named_steps['model'], X.columns, name, model_type)
                        
        return results, predictions
        
    def plot_model_comparisons(self, results, predictions, model_type='absolute'):
        """
        Plot comparison of model performance with CV metrics - fixed metric names
        """
        eval_subdir = os.path.join(self.eval_dir, model_type)
        
        metrics_df = pd.DataFrame({
            model: {
                'Train RMSE (mean)': results[model]['train_rmse_mean'],
                'Train RMSE (std)': results[model]['train_rmse_std'],
                'Test RMSE (mean)': results[model]['test_rmse_mean'],
                'Test RMSE (std)': results[model]['test_rmse_std'],
                'Train MAE': results[model]['train_mae_mean'],
                'Test MAE': results[model]['test_mae_mean'],
                'Train R²': results[model]['train_r2_mean'],
                'Test R²': results[model]['test_r2_mean']
            }
            for model in results.keys()
        }).T
        
        metrics_df.to_csv(os.path.join(eval_subdir, 'model_performance_cv.csv'))
        
        plt.figure(figsize=(12, 6))
        plt.errorbar(metrics_df.index, metrics_df['Train RMSE (mean)'], 
                    yerr=metrics_df['Train RMSE (std)'], label='Train RMSE', fmt='o')
        plt.errorbar(metrics_df.index, metrics_df['Test RMSE (mean)'], 
                    yerr=metrics_df['Test RMSE (std)'], label='Test RMSE', fmt='o')
        plt.title(f'{self.target_fuel} - Model RMSE Comparison ({model_type})')
        plt.ylabel('RMSE')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_subdir, 'rmse_comparison_cv.png'))
        plt.close()
        
        plt.figure(figsize=(12, 6))
        metrics_df[['Train R²', 'Test R²']].plot(kind='bar')
        plt.title(f'{self.target_fuel} - Model R² Comparison ({model_type})')
        plt.ylabel('R²')
        plt.grid(True, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(eval_subdir, 'r2_comparison.png'))
        plt.close()
        
        best_model = metrics_df['Test R²'].idxmax()
        plt.figure(figsize=(15, 6))
        
        plt.subplot(1, 2, 1)
        plt.scatter(predictions[best_model]['y_train'], predictions[best_model]['y_pred_train'])
        plt.plot([min(predictions[best_model]['y_train']), max(predictions[best_model]['y_train'])], 
                [min(predictions[best_model]['y_train']), max(predictions[best_model]['y_train'])], 
                'k--')
        plt.title(f'Training Data: Actual vs Predicted - {best_model}')
        plt.xlabel('Actual')
        plt.ylabel('Predicted')
        plt.grid(True)
        
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
        plt.savefig(os.path.join(eval_subdir, 'best_model_predictions.png'))
        plt.close()
        
        plt.figure(figsize=(12, 6))
        plt.plot(predictions[best_model]['y_test'].index, predictions[best_model]['y_test'], 'b-', label='Actual')
        plt.plot(predictions[best_model]['y_test'].index, predictions[best_model]['y_pred_test'], 'r--', label='Predicted')
        plt.title(f'{self.target_fuel} - Test Data Predictions ({best_model})')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_subdir, 'test_predictions_timeseries.png'))
        plt.close()
        
    def plot_feature_importance(self, model, feature_names, model_name, model_type='absolute'):
        """
        Plot feature importance for tree-based models
        
        Args:
            model: Trained model
            feature_names: Names of features
            model_name: Name of the model
            model_type (str): Type of model ('absolute' or 'percent_change')
        """
        # Get subdirectory for evaluation outputs
        eval_subdir = os.path.join(self.eval_dir, model_type, 'feature_importance')
        ensure_directory(eval_subdir)
        
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
            plt.title(f'{self.target_fuel} - Feature Importance - {model_name}')
            plt.bar(range(top_n), importance[indices][:top_n], align='center')
            plt.xticks(range(top_n), [feature_names[i] for i in indices][:top_n], rotation=90)
            plt.tight_layout()
            plt.savefig(os.path.join(eval_subdir, f'{model_name.lower().replace(" ", "_")}.png'))
            plt.close()
            
            # Save importance to CSV for reference
            feature_importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importance
            }).sort_values('Importance', ascending=False)
            
            feature_importance_df.to_csv(
                os.path.join(eval_subdir, f'{model_name.lower().replace(" ", "_")}.csv'),
                index=False
            )
            
        except Exception as e:
            logger.error(f"Error plotting feature importance: {str(e)}")
            
    def optimize_model(self, model_type, X, y, test_size=12, param_grid=None, data_type='absolute'):
        """
        Optimize hyperparameters for a specific model
        
        Args:
            model_type (str): Type of model to optimize
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            test_size (int): Number of months to use for testing
            param_grid (dict): Parameter grid for GridSearchCV
            data_type (str): Type of data ('absolute' or 'percent_change')
            
        Returns:
            tuple: Best model, best parameters, CV results
        """
        # Create subdirectories for optimized models
        model_subdir = ensure_directory(os.path.join(self.model_dir, data_type, 'optimized'))
        eval_subdir = ensure_directory(os.path.join(self.eval_dir, data_type, 'optimized'))
        
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
        logger.info(f"Optimizing {model_type} model for {self.target_fuel} using {data_type} data...")
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
        model_path = os.path.join(model_subdir, f"{model_type}.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(best_model, f)
        
        # Plot feature importance if applicable
        if model_type in ['randomforest', 'gbm', 'xgboost']:
            self.plot_feature_importance(
                best_model, 
                X.columns, 
                f"{model_type}_optimized", 
                data_type
            )
            
        # Plot predictions
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, 'b-', label='Actual')
        plt.plot(y_test.index, y_pred, 'r--', label='Predicted')
        plt.title(f'{self.target_fuel} - Optimized {model_type.capitalize()} Predictions ({data_type})')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_subdir, f'{model_type}_predictions.png'))
        plt.close()
        
        return best_model, best_params, grid_search.cv_results_
        
    def ensemble_prediction(self, X, y, test_size=12, models_to_use=None, data_type='absolute'):
        """
        Create an ensemble prediction by averaging results from multiple models
        
        Args:
            X (pd.DataFrame): Feature data
            y (pd.Series): Target data
            test_size (int): Number of months to use for testing
            models_to_use (list): List of model types to include in ensemble
            data_type (str): Type of data ('absolute' or 'percent_change')
            
        Returns:
            dict: Dictionary with ensemble results
        """
        # Create subdirectories for ensemble
        model_subdir = ensure_directory(os.path.join(self.model_dir, data_type, 'ensemble'))
        eval_subdir = ensure_directory(os.path.join(self.eval_dir, data_type, 'ensemble'))
        
        # Split data
        X_train, X_test, y_train, y_test = self.train_test_split(X, y, test_size)
        
        # Default models if not specified
        if models_to_use is None:
            models_to_use = ['elasticnet', 'randomforest', 'gbm', 'xgboost']
            
        # Train individual models
        trained_models = {}
        predictions = {}
        
        for model_type in models_to_use:
            # Try to load optimized model first
            optimized_path = os.path.join(self.model_dir, data_type, 'optimized', f"{model_type}.pkl")
            
            if os.path.exists(optimized_path):
                logger.info(f"Loading optimized {model_type} model...")
                with open(optimized_path, 'rb') as f:
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
        
        logger.info(f"Ensemble model for {self.target_fuel} ({data_type}) - Test RMSE: {ensemble_rmse:.4f}, Test R²: {ensemble_r2:.4f}")
        
        # Plot ensemble results
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test, 'b-', label='Actual')
        plt.plot(y_test.index, ensemble_pred, 'r--', label='Ensemble Predicted')
        
        # Add individual model predictions
        for model_type in models_to_use:
            plt.plot(y_test.index, predictions[model_type], '--', alpha=0.3, label=f'{model_type.capitalize()} Predicted')
            
        plt.title(f'{self.target_fuel} - Ensemble Model Predictions ({data_type})')
        plt.xlabel('Date')
        plt.ylabel('Value')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(os.path.join(eval_subdir, 'predictions.png'))
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
        
        with open(os.path.join(model_subdir, 'results.pkl'), 'wb') as f:
            pickle.dump(ensemble_results, f)
            
        return ensemble_results

def main():
    try:
        logger.info("Starting model building process...")
        
        # Build models for Diesel demand
        logger.info("Building models for Diesel demand...")
        diesel_forecaster = FuelDemandForecaster(target_fuel='Diesel')
        
        # Prepare data for absolute values
        X_diesel, y_diesel = diesel_forecaster.prepare_data(
            use_pct_change=False,
            lag_features=True,
            max_lag=3,
            add_seasonality=True
        )
        
        # Build and evaluate models
        diesel_results, diesel_predictions = diesel_forecaster.build_models(
            X_diesel, y_diesel, test_size=12, model_type='absolute'
        )
        
        # Optimize the best performing model type based on CV test R² mean
        best_model_diesel = max(diesel_results.items(), key=lambda x: x[1]['test_r2_mean'])[0]
        
        if best_model_diesel == 'Linear Regression':
            model_type = 'elasticnet'
        elif best_model_diesel == 'Random Forest':
            model_type = 'randomforest'
        elif best_model_diesel == 'Gradient Boosting':
            model_type = 'gbm'
        elif best_model_diesel == 'XGBoost':
            model_type = 'xgboost'
        else:
            model_type = 'elasticnet'
            
        logger.info(f"Optimizing {model_type} model for Diesel demand...")
        diesel_forecaster.optimize_model(model_type, X_diesel, y_diesel, data_type='absolute')
        
        # Try percent change model
        logger.info("Building percent change models for Diesel demand...")
        X_diesel_pct, y_diesel_pct = diesel_forecaster.prepare_data(
            use_pct_change=True,
            lag_features=True,
            max_lag=3,
            add_seasonality=True
        )
        
        diesel_pct_results, _ = diesel_forecaster.build_models(
            X_diesel_pct, y_diesel_pct, test_size=12, model_type='percent_change'
        )
        
        # Build ensemble model
        logger.info("Building ensemble model for Diesel demand...")
        diesel_forecaster.ensemble_prediction(X_diesel, y_diesel, data_type='absolute')
        
        # Repeat for Gasoline demand
        logger.info("Building models for Gasoline demand...")
        gasoline_forecaster = FuelDemandForecaster(target_fuel='Gasoline')
        
        X_gasoline, y_gasoline = gasoline_forecaster.prepare_data(
            use_pct_change=False,
            lag_features=True,
            max_lag=3,
            add_seasonality=True
        )
        
        gasoline_results, gasoline_predictions = gasoline_forecaster.build_models(
            X_gasoline, y_gasoline, test_size=12, model_type='absolute'
        )
        
        # Optimize the best performing model type based on CV test R² mean
        best_model_gasoline = max(gasoline_results.items(), key=lambda x: x[1]['test_r2_mean'])[0]
        
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
        gasoline_forecaster.optimize_model(model_type, X_gasoline, y_gasoline, data_type='absolute')
        
        logger.info("Building percent change models for Gasoline demand...")
        X_gasoline_pct, y_gasoline_pct = gasoline_forecaster.prepare_data(
            use_pct_change=True,
            lag_features=True,
            max_lag=3,
            add_seasonality=True
        )
        
        gasoline_pct_results, _ = gasoline_forecaster.build_models(
            X_gasoline_pct, y_gasoline_pct, test_size=12, model_type='percent_change'
        )
        
        logger.info("Building ensemble model for Gasoline demand...")
        gasoline_forecaster.ensemble_prediction(X_gasoline, y_gasoline, data_type='absolute')
        
        logger.info("Model building process completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in model building process: {str(e)}", exc_info=True)

if __name__=="__main__":
    main()