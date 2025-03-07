"""
Configuration loader for fuel demand analysis

This module handles loading and validating the JSON configuration file
that controls the analysis pipeline.
"""

import json
import os
import logging
from datetime import datetime
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

def load_config(config_path: str = "config.json") -> Dict[str, Any]:
    """
    Load and validate the configuration file
    
    Args:
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Configuration dictionary
    """
    # Check if config file exists
    if not os.path.exists(config_path):
        logger.warning(f"Config file {config_path} not found. Using default configuration.")
        return get_default_config()
    
    # Load config file
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        logger.info(f"Successfully loaded configuration from {config_path}")
        
        # Validate and set defaults for missing values
        config = validate_config(config)
        return config
    
    except Exception as e:
        logger.error(f"Error loading config file: {str(e)}")
        logger.warning("Using default configuration instead.")
        return get_default_config()

def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate configuration and set defaults for missing values
    
    Args:
        config (Dict[str, Any]): Configuration dictionary
        
    Returns:
        Dict[str, Any]: Validated configuration with defaults applied
    """
    default_config = get_default_config()
    
    # Helper function to recursively merge dictionaries
    def merge_dicts(source, destination):
        for key, value in source.items():
            if key not in destination:
                destination[key] = value
            elif isinstance(value, dict) and isinstance(destination[key], dict):
                merge_dicts(value, destination[key])
        return destination
    
    # Merge with defaults to ensure all required fields exist
    validated_config = merge_dicts(default_config, config)
    
    # Convert date strings to datetime objects
    if validated_config['data']['date_range']['start_date']:
        try:
            datetime.strptime(validated_config['data']['date_range']['start_date'], '%Y-%m-%d')
        except ValueError:
            logger.warning(f"Invalid start_date format. Using default instead.")
            validated_config['data']['date_range']['start_date'] = default_config['data']['date_range']['start_date']
    
    if validated_config['data']['date_range']['end_date']:
        try:
            datetime.strptime(validated_config['data']['date_range']['end_date'], '%Y-%m-%d')
        except ValueError:
            logger.warning(f"Invalid end_date format. Using default instead.")
            validated_config['data']['date_range']['end_date'] = default_config['data']['date_range']['end_date']
    
    return validated_config

def get_default_config() -> Dict[str, Any]:
    """
    Get default configuration
    
    Returns:
        Dict[str, Any]: Default configuration dictionary
    """
    return {
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
            "charts_to_generate": ["trends", "seasonality", "correlations", "feature_importance", "predictions", "pca"],
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

def update_config(updates: Dict[str, Any], config_path: str = "config.json") -> Dict[str, Any]:
    """
    Update configuration file with new values
    
    Args:
        updates (Dict[str, Any]): Dictionary of updates to apply
        config_path (str): Path to configuration file
        
    Returns:
        Dict[str, Any]: Updated configuration
    """
    # Load current config
    current_config = load_config(config_path)
    
    # Helper function to recursively update dictionaries
    def update_dict(source, updates_dict):
        for key, value in updates_dict.items():
            if key in source and isinstance(value, dict) and isinstance(source[key], dict):
                update_dict(source[key], value)
            else:
                source[key] = value
        return source
    
    # Apply updates
    updated_config = update_dict(current_config, updates)
    
    # Save updated config
    try:
        with open(config_path, 'w') as f:
            json.dump(updated_config, f, indent=2)
        logger.info(f"Updated configuration saved to {config_path}")
    except Exception as e:
        logger.error(f"Error saving updated configuration: {str(e)}")
    
    return updated_config

def create_default_config(config_path: str = "config.json") -> None:
    """
    Create a default configuration file if one doesn't exist
    
    Args:
        config_path (str): Path where configuration file should be created
    """
    if not os.path.exists(config_path):
        try:
            with open(config_path, 'w') as f:
                json.dump(get_default_config(), f, indent=2)
            logger.info(f"Created default configuration file at {config_path}")
        except Exception as e:
            logger.error(f"Error creating default configuration file: {str(e)}")