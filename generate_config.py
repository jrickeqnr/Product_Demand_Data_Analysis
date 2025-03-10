#!/usr/bin/env python
"""
Generate default configuration file for the Fuel Demand Analysis project.
This script creates a config.json file with default settings if it doesn't exist.
"""

import json
import os
import logging
import argparse
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

def generate_default_config():
    """Generate the default configuration dictionary"""
    
    config = {
        "data": {
            "date_range": {
                "start_date": "2015-01-01",
                "end_date": None
            },
            "data_frequency": "monthly",
            "fill_missing_strategy": "median"
        },
        "analysis": {
            "models_to_run": ["absolute", "percent_change"],
            "feature_selection": {
                "enabled": False,
                "method": "correlation",
                "top_features": 20
            },
            "lag_features": {
                "enabled": True,
                "max_lag": 3
            },
            "seasonality": {
                "enabled": True,
                "add_month_dummies": True,
                "add_quarter_dummies": True
            },
            "rolling_window": {
                "window": 72
            },
            "pca": {
                "enabled": True,
                "variance_threshold": 0.9
            }
        },
        "modeling": {
            "train_test_split": {
                "test_size": 12
            },
            "models": {
                "linear_regression": True,
                "elasticnet": True,
                "random_forest": True,
                "gradient_boosting": True,
                "xgboost": True
            },
            "hyperparameter_optimization": {
                "enabled": True,
                "cv_folds": 5,
                "models_to_optimize": ["elasticnet", "randomforest", "gbm", "xgboost"]
            },
            "ensemble": {
                "enabled": True,
                "models_to_include": ["elasticnet", "randomforest", "gbm", "xgboost"]
            }
        },
        "targets": {
            "diesel": True,
            "gasoline": True
        },
        "visualization": {
            "charts_to_generate": ["trends", "seasonality", "correlations", "feature_importance", 
                                "predictions", "pca", "correlation_evolution", "correlation_table"],
            "save_formats": ["png"],
            "dpi": 300
        },
        "logging": {
            "level": "INFO",
            "file_logging": True,
            "console_logging": True
        },
        "paths": {
            "data_dir": "data",
            "charts_dir": "charts",
            "models_dir": "models",
            "evaluation_dir": "evaluation"
        }
    }
    
    return config

def save_config(config, filepath="config.json", overwrite=False):
    """Save configuration to a JSON file
    
    Args:
        config (dict): Configuration dictionary
        filepath (str): Path to save the config file
        overwrite (bool): Whether to overwrite existing file
    
    Returns:
        bool: True if file was saved, False otherwise
    """
    if os.path.exists(filepath) and not overwrite:
        logger.warning(f"Config file {filepath} already exists. Use --overwrite to replace it.")
        return False
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)
    
    logger.info(f"Configuration saved to {filepath}")
    return True

def main():
    parser = argparse.ArgumentParser(description='Generate default configuration file for Fuel Demand Analysis')
    parser.add_argument('--output', default='config.json', help='Output file path (default: config.json)')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing file if it exists')
    
    args = parser.parse_args()
    
    # Generate the default configuration
    config = generate_default_config()
    
    # Save the configuration
    if save_config(config, args.output, args.overwrite):
        logger.info("Configuration file generated successfully.")
    else:
        logger.info("No changes made to existing configuration file.")

if __name__ == "__main__":
    main()