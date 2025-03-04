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

def run_script(script_name, description):
    """Run a Python script and handle any errors"""
    logger.info(f"Starting {description}...")
    start_time = time.time()
    
    try:
        result = subprocess.run([sys.executable, script_name], 
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