import os
import pandas as pd
from data_fetcher import SynapseConnector, BloombergDataFetcher, EIADataFetcher, DataProcessor
import pickle
import logging
import time
import numpy as np
import argparse
from datetime import datetime
from config_loader import load_config

def setup_logging(config):
    """Set up logging based on configuration"""
    log_level = getattr(logging, config['logging']['level'])
    handlers = []
    
    if config['logging']['file_logging']:
        handlers.append(logging.FileHandler("data_fetching.log"))
    
    if config['logging']['console_logging']:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Fetch and process fuel demand data')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    global logger
    logger = setup_logging(config)
    
    # Create data directory if it doesn't exist
    data_dir = config['paths']['data_dir']
    os.makedirs(data_dir, exist_ok=True)
    
    start_date = config['data']['date_range']['start_date']
    end_date = config['data']['date_range']['end_date']
    
    try:
        logger.info("Starting data fetching process...")
        logger.info(f"Date range: {start_date} to {end_date or 'present'}")
        
        # Initialize connector
        connector = SynapseConnector()
        
        # Fetch Bloomberg data
        logger.info("Fetching Bloomberg data...")
        bloomberg_fetcher = BloombergDataFetcher(connector)
        bloomberg_raw_data = bloomberg_fetcher.fetch_all_data()
        
        # Check if Bloomberg data was retrieved successfully
        if bloomberg_raw_data.empty:
            logger.error("Failed to retrieve Bloomberg data. Exiting process.")
            return
        
        # Apply date filtering if configured
        if start_date or end_date:
            logger.info(f"Filtering data by date range: {start_date} to {end_date or 'present'}")
            bloomberg_raw_data['Date'] = pd.to_datetime(bloomberg_raw_data['Date'])
            
            if start_date:
                bloomberg_raw_data = bloomberg_raw_data[bloomberg_raw_data['Date'] >= start_date]
                
            if end_date:
                bloomberg_raw_data = bloomberg_raw_data[bloomberg_raw_data['Date'] <= end_date]
        
        # Save raw Bloomberg data first
        bloomberg_raw_data.to_csv(os.path.join(data_dir, 'bloomberg_raw_data.csv'), index=False)
        logger.info(f"Saved raw Bloomberg data with {len(bloomberg_raw_data)} rows")
        
        # Aggregate Bloomberg data to monthly frequency with retry logic
        max_retries = 3
        retry_count = 0
        bloomberg_monthly_data = None
        
        while retry_count < max_retries:
            try:
                logger.info(f"Attempting to aggregate Bloomberg data (attempt {retry_count + 1}/{max_retries})...")
                bloomberg_monthly_data = bloomberg_fetcher.aggregate_to_monthly(bloomberg_raw_data)
                
                if not bloomberg_monthly_data.empty:
                    break
                else:
                    logger.warning("Empty result from aggregation attempt")
                    retry_count += 1
                    time.sleep(2)
            except Exception as e:
                logger.warning(f"Aggregation attempt {retry_count + 1} failed: {str(e)}")
                retry_count += 1
                time.sleep(2)
        
        # Check if aggregation was successful
        if bloomberg_monthly_data is None or bloomberg_monthly_data.empty:
            logger.error("Failed to aggregate Bloomberg data after multiple attempts. Exiting process.")
            return
        
        # Save Bloomberg monthly data
        bloomberg_monthly_data.to_csv(os.path.join(data_dir, 'bloomberg_monthly_data.csv'), index=False)
        logger.info(f"Saved Bloomberg monthly data with {len(bloomberg_monthly_data)} rows")
        
        # Fetch EIA fuel demand data with retry logic
        logger.info("Fetching EIA fuel demand data...")
        eia_fetcher = EIADataFetcher(connector)
        
        # Try up to 3 times to get the EIA data
        max_retries = 3
        retry_count = 0
        fuel_demand_data = None
        
        while retry_count < max_retries:
            try:
                # Explicitly reconnect for each attempt for EIA data
                connector.connect()
                time.sleep(1)
                
                logger.info(f"EIA data fetch attempt {retry_count + 1}/{max_retries}...")
                fuel_demand_data = eia_fetcher.fetch_fuel_demand_data()
                
                if not fuel_demand_data.empty:
                    logger.info(f"Successfully retrieved EIA data with {len(fuel_demand_data)} rows")
                    break
                else:
                    logger.warning("Empty result from EIA data fetch")
                    retry_count += 1
                    time.sleep(3)
            except Exception as e:
                logger.warning(f"EIA data fetch attempt {retry_count + 1} failed: {str(e)}")
                retry_count += 1
                time.sleep(3)
        
        # If all attempts fail, try using direct query through SynapseConnector
        if fuel_demand_data is None or fuel_demand_data.empty:
            logger.warning("All EIA fetcher attempts failed. Trying direct query...")
            
            try:
                # Force a fresh connection
                connector.connect()
                time.sleep(2)
                
                query = """
                SELECT 
                    date,
                    SUM(CASE WHEN series_id IN ('PET.MD0UP_NUS_2.M', 'PET.M_EPOORXFE_VPP_NUS_MBBLD.M') THEN value ELSE 0 END) as Diesel,
                    SUM(CASE WHEN series_id = 'PET.MGFUPUS2.M' THEN value ELSE 0 END) as Gasoline
                FROM [atlas].[standard_eiapetroleum_timeseries_v3latest]
                WHERE series_id IN ('PET.MD0UP_NUS_2.M', 'PET.M_EPOORXFE_VPP_NUS_MBBLD.M', 'PET.MGFUPUS2.M')
                GROUP BY date
                ORDER BY date ASC;
                """
                
                fuel_demand_data = connector.query_to_df(query)
                
                if fuel_demand_data is not None and not fuel_demand_data.empty:
                    fuel_demand_data['date'] = pd.to_datetime(fuel_demand_data['date'])
                    logger.info(f"Direct query successful, retrieved {len(fuel_demand_data)} rows of EIA data")
                else:
                    logger.error("Direct query for EIA data failed as well")
                    # Create an empty DataFrame with the right structure
                    fuel_demand_data = pd.DataFrame(columns=['date', 'Diesel', 'Gasoline'])
            except Exception as e:
                logger.error(f"Direct query attempt failed: {str(e)}")
                fuel_demand_data = pd.DataFrame(columns=['date', 'Diesel', 'Gasoline'])
        
        # Apply date filtering if configured
        if start_date or end_date:
            fuel_demand_data['date'] = pd.to_datetime(fuel_demand_data['date'])
            
            if start_date:
                fuel_demand_data = fuel_demand_data[fuel_demand_data['date'] >= start_date]
                
            if end_date:
                fuel_demand_data = fuel_demand_data[fuel_demand_data['date'] <= end_date]
        
        # Save fuel demand data
        fuel_demand_data.to_csv(os.path.join(data_dir, 'fuel_demand_data.csv'), index=False)
        logger.info(f"Saved fuel demand data with {len(fuel_demand_data)} rows")
        
        # Preprocess and combine data
        logger.info("Processing and combining data...")
        try:
            input_pivot, input_pivot_pct, demand_absolute, demand_pct, category_map = DataProcessor.prepare_data(
                bloomberg_monthly_data, fuel_demand_data)
            
            # Save processed data
            input_pivot.to_csv(os.path.join(data_dir, 'input_features_absolute.csv'))
            input_pivot_pct.to_csv(os.path.join(data_dir, 'input_features_percent_change.csv'))
            demand_absolute.to_csv(os.path.join(data_dir, 'demand_absolute.csv'))
            demand_pct.to_csv(os.path.join(data_dir, 'demand_percent_change.csv'))
            
            # Save category map for later reference
            with open(os.path.join(data_dir, 'category_map.pkl'), 'wb') as f:
                pickle.dump(category_map, f)
                
            logger.info("Data processing completed successfully.")
            
            # Print some statistics about the data
            try:
                if not input_pivot.empty:
                    logger.info(f"Input features time range: {input_pivot.index.min()} to {input_pivot.index.max()}")
                    logger.info(f"Number of input features: {input_pivot.shape[1]}")
                
                if not demand_absolute.empty:
                    logger.info(f"Demand data time range: {demand_absolute.index.min()} to {demand_absolute.index.max()}")
                else:
                    logger.warning("Demand data is empty, no statistics to report")
            except Exception as stats_e:
                logger.warning(f"Error when generating statistics: {str(stats_e)}")
                
        except Exception as e:
            logger.error(f"Error in data processing: {str(e)}", exc_info=True)
        
    except Exception as e:
        logger.error(f"Error in data fetching process: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()