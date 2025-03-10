"""
Fuel Demand Analysis and Forecasting - Main Script

This script coordinates the execution of the entire workflow:
1. Data fetching
2. Exploratory data analysis
3. Model building
"""

import os
import logging
import argparse
import time
from datetime import datetime
import subprocess
import sys
import json

# Configure logging
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(log_dir, f"main_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

def ensure_config_exists():
    """Check if config.json exists and generate it if not"""
    config_path = 'config.json'
    
    if not os.path.exists(config_path):
        logger.info("Configuration file not found. Generating default configuration...")
        if run_script('generate_config.py', 'Configuration generation'):
            logger.info("Default configuration generated successfully.")
            return True
        else:
            logger.error("Failed to generate configuration file.")
            return False
    else:
        # Validate that the config file is proper JSON
        try:
            with open(config_path, 'r') as f:
                json.load(f)
            logger.info("Configuration file exists and is valid.")
            return True
        except json.JSONDecodeError:
            logger.error("Configuration file exists but contains invalid JSON.")
            logger.info("Generating new default configuration...")
            if run_script('generate_config.py', 'Configuration generation', ['--overwrite']):
                logger.info("Default configuration generated successfully.")
                return True
            else:
                logger.error("Failed to generate configuration file.")
                return False

def run_script(script_name, description, additional_args=None):
    """Run a Python script and handle any errors"""
    logger.info(f"Starting {description}...")
    start_time = time.time()
    
    cmd = [sys.executable, script_name]
    if additional_args:
        cmd.extend(additional_args)
    
    try:
        result = subprocess.run(cmd, 
                               check=True, 
                               stdout=subprocess.PIPE, 
                               stderr=subprocess.PIPE,
                               universal_newlines=True)
        
        logger.info(f"{description} completed successfully in {time.time() - start_time:.2f} seconds")
        return True
    
    except subprocess.CalledProcessError as e:
        logger.error(f"{description} failed with exit code {e.returncode}")
        logger.error(f"STDOUT: {e.stdout}")
        logger.error(f"STDERR: {e.stderr}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Fuel Demand Analysis and Forecasting')
    parser.add_argument('--skip-fetch', action='store_true', 
                        help='Skip data fetching step (use existing data)')
    parser.add_argument('--skip-eda', action='store_true',
                        help='Skip exploratory data analysis step')
    parser.add_argument('--skip-model', action='store_true',
                        help='Skip model building step')
    parser.add_argument('--regenerate-config', action='store_true',
                        help='Regenerate configuration file with default settings')
    
    args = parser.parse_args()
    
    # Welcome message
    logger.info("=" * 80)
    logger.info("Fuel Demand Analysis and Forecasting Workflow")
    logger.info("=" * 80)
    
    # Create necessary directories
    os.makedirs('data', exist_ok=True)
    os.makedirs('charts', exist_ok=True)
    os.makedirs('models', exist_ok=True)
    os.makedirs('evaluation', exist_ok=True)
    
    # Step 0: Check/generate configuration
    if args.regenerate_config:
        logger.info("Regenerating configuration as requested...")
        if not run_script('generate_config.py', 'Configuration generation', ['--overwrite']):
            logger.error("Failed to regenerate configuration. Stopping workflow.")
            return
    else:
        if not ensure_config_exists():
            logger.error("Cannot proceed without valid configuration. Stopping workflow.")
            return
    
    # Step 1: Fetch data
    if not args.skip_fetch:
        if not run_script('fetch_data.py', 'Data fetching'):
            logger.error("Data fetching failed. Stopping workflow.")
            return
    else:
        logger.info("Skipping data fetching as requested")
    
    # Step 2: Exploratory data analysis
    if not args.skip_eda:
        if not run_script('exploratory_analysis.py', 'Exploratory data analysis'):
            logger.warning("Exploratory data analysis failed. Continuing with model building.")
    else:
        logger.info("Skipping exploratory data analysis as requested")
    
    # Step 3: Model building
    if not args.skip_model:
        if not run_script('model_builder.py', 'Model building'):
            logger.error("Model building failed.")
            return
    else:
        logger.info("Skipping model building as requested")
    
    logger.info("=" * 80)
    logger.info("Workflow completed")
    logger.info("=" * 80)
    
    # Print summary of generated artifacts
    if os.path.exists('data'):
        data_files = os.listdir('data')
        logger.info(f"Generated data files: {len(data_files)}")
    
    if os.path.exists('charts'):
        chart_files = os.listdir('charts')
        logger.info(f"Generated chart files: {len(chart_files)}")
    
    if os.path.exists('models'):
        model_files = os.listdir('models')
        logger.info(f"Generated model files: {len(model_files)}")
    
    if os.path.exists('evaluation'):
        eval_files = os.listdir('evaluation')
        logger.info(f"Generated evaluation files: {len(eval_files)}")
    
    logger.info("To explore the results interactively, run: jupyter notebook interactive_exploration.ipynb")

if __name__ == "__main__":
    main()