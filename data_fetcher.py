import pandas as pd
import logging
from datetime import datetime
from typing import Optional, Dict, List, Tuple
import numpy as np
import time

class SynapseConnector:
    def __init__(
        self,
        server: str = 'eqnr-atlas-prod-syn-xxg-ondemand.sql.azuresynapse.net',
        database: str = 'atlas_ondemand',
        user_id: str = 'jrick@equinor.com',
        authentication: str = 'ActiveDirectoryInteractive'
    ):
        self.server = server
        self.database = database
        self.user_id = user_id
        self.authentication = authentication
        self.connection = None
        self.sa_engine = None
        
        # Set up logging
        self.logger = logging.getLogger(__name__)

    def connect(self) -> None:
        """
        Establishes connection to Synapse database.
        Will prompt for authentication in browser window.
        """
        try:
            import pyodbc
            import sqlalchemy as sa
            
            # Clear any existing connection
            if self.connection:
                try:
                    self.connection.close()
                except:
                    pass
                self.connection = None
                
            if self.sa_engine:
                try:
                    self.sa_engine.dispose()
                except:
                    pass
                self.sa_engine = None
            
            connection_string = (
                'DRIVER={ODBC Driver 18 for SQL Server};'
                f'SERVER={self.server};'
                f'DATABASE={self.database};'
                f'Authentication={self.authentication};'
                f'UID={self.user_id};'
                'TrustServerCertificate=no'
            )
            
            self.connection = pyodbc.connect(connection_string)
            
            # Use a more robust way to create the SQLAlchemy engine
            try:
                from sqlalchemy.engine import URL
                connection_url = URL.create(
                    "mssql+pyodbc", 
                    query={"odbc_connect": connection_string}
                )
                self.sa_engine = sa.create_engine(connection_url)
            except ImportError:
                # Fallback for older SQLAlchemy versions
                self.sa_engine = sa.create_engine('mssql+pyodbc://', creator=lambda: self.connection)
                
            self.logger.info("Successfully connected to Synapse database")
            
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {str(e)}")
            raise

    def query_to_df(self, query: str) -> Optional[pd.DataFrame]:
        """
        Executes SQL query and returns results as pandas DataFrame.
        
        Args:
            query (str): SQL query to execute
            
        Returns:
            Optional[pd.DataFrame]: Query results as DataFrame, None if query fails
        """
        try:
            if not self.connection or not self.sa_engine:
                self.connect()
            
            # Add retry logic for query execution
            max_retries = 3
            retry_count = 0
            
            while retry_count < max_retries:
                try:
                    df = pd.read_sql(query, self.sa_engine)
                    self.logger.info(f"Query executed successfully. Returned {len(df)} rows")
                    return df
                except Exception as e:
                    retry_count += 1
                    if "closed automatically" in str(e) and retry_count < max_retries:
                        self.logger.warning(f"Query attempt {retry_count} failed. Reconnecting and retrying...")
                        self.connect()  # Reconnect before retrying
                        time.sleep(2)
                    else:
                        raise e
            
        except Exception as e:
            self.logger.error(f"Query execution failed: {str(e)}")
            return None


class BloombergDataFetcher:
    def __init__(self, synapse_connector):
        self.connector = synapse_connector
        self.logger = logging.getLogger(__name__)
        
        # Dictionary mapping categories to Bloomberg codes
        self.series_mapping = {
            'PMI': [
                ('ISM Manufacturing', 'NAPMPMI Index'),
                ('ISM Manufacturing Employment', 'NAPMEMPL Index'),
                ('ISM Services Employment', 'NAPMNEMP Index'),
                ('ISM Composite', 'NAPMALL Index')
            ],
            'Housing': [
                ('S&P Case Shiller Home Index', 'SPCS20 Index')
            ],
            'Financial': [
                ('S&P 500 Index', 'SPX Index'),
                ('Fed Funds Rate', 'FEDL01 Index')
            ],
            'Industrial': [
                ('Raw Steel Production', 'ST23ST Index'),
                ('Industrial Production SA', 'IP Index')
            ],
            'Economic': [
                ('UM Consumer Sentiment', 'CONSSENT Index'),
                ('Auto Sales', 'SAARTOTL Index'),
                ('Auto Inventory Level SA', 'CAR INV Index'),
                ('Disposable Personal Income (Real)', 'PIDSDCWT Index'),
                ('Disposable Personal Income (Nominal)', 'PITL Index'),
                ('Retail Sales', 'MTSLRRT$ Index'),
                ('GDP', 'GDP CUR$ Index'),
                ('GDP - Real', 'VADRALLI Index'),
                ('Personal Consumption of Non-Durable Goods', 'PCE NDRB Index'),
                ('Producer Price Index', 'PCAC Index')
            ],
            'Trade': [
                ('Baltic Dry Index', 'BDIY Index')
            ],
            'Transport': [
                ('Cass Corp Freight Index', 'CASSSHIP Index'),
                ('Vehicle Miles Travelled SA', 'VMTDTRSA Index'),
                ('Airline Load Factor', 'ARSALDFC Index'),
                ('Airline Seat Miles', 'ARSAASML Index'),
                ('Number of Flights', 'ARSAFLGH Index'),
                ('ATA Truck Tonnage Index', 'TTINSATV Index')
            ],
            'Labor': [
                ('Unemployment Rate (U-3)', 'USURTOT Index'),
                ('Unemployment Rate (U-6)', 'USUDMAER Index'),
                ('Continuing Claims', 'INJCSPNS Index'),
                ('Initial Jobless Claims', 'INJCJC Index'),
                ('Avg. Weekly Hours (All private)', 'AWH TOTL Index'),
                ('Weekly Payroll', 'USAPTOTN Index'),
                ('Weekly Wages', 'WEWSFCKD Index')
            ],
            'Weather': [
                ('Heating Degree Days', 'NOAHHT Index'),
                ('Cooling Degree Days', 'NOACMCT Index')
            ],
            'Price': [
                ('U.S. Gasoline - Average Retail', 'NGAPGRPG Index'),
                ('U.S. Diesel - Average Retail', 'NGAPGRPD Index')
            ]
        }
        
        # Create a reverse mapping for quick lookup
        self.ticker_to_info = {
            ticker: (variable, category)
            for category, series_list in self.series_mapping.items()
            for variable, ticker in series_list
        }

    def fetch_all_data(self) -> pd.DataFrame:
        """
        Fetch data for all series in a single query and process into a DataFrame
        """
        # Get all tickers
        all_tickers = [ticker for series_list in self.series_mapping.values() 
                      for _, ticker in series_list]
        
        # Create a single query with all tickers
        tickers_str = "', '".join(all_tickers)
        query = f"""
        SELECT 
            lastUpdateDt AS date,
            recordIdentifier AS Ticker,
            Value,
            indxFreq AS Frequency
        FROM [atlas].[standard_bloombergdl_summary_v2latest]
        WHERE recordIdentifier IN ('{tickers_str}')
        AND etl_dataset = 'historical'
        AND Profile = 'pxLast'
        ORDER BY lastUpdateDt
        """
        
        try:
            df = self.connector.query_to_df(query)
            if df is not None and not df.empty:
                self.logger.info(f"Retrieved {len(df)} rows of data for {len(all_tickers)} tickers")
                
                # Add Category and Variable columns using the reverse mapping
                df['Variable'] = df['Ticker'].map(lambda x: self.ticker_to_info.get(x, ('Unknown', 'Unknown'))[0])
                df['Category'] = df['Ticker'].map(lambda x: self.ticker_to_info.get(x, ('Unknown', 'Unknown'))[1])
                
                # Drop the temporary Ticker column and ensure Date is datetime
                df = df.drop('Ticker', axis=1)
                df['date'] = pd.to_datetime(df['date'])
                
                return df[['date', 'Category', 'Variable', 'Value', 'Frequency']]
            else:
                self.logger.warning("No data retrieved from the query")
                # Return an empty DataFrame with the correct structure
                return pd.DataFrame(columns=['date', 'Category', 'Variable', 'Value', 'Frequency'])
                
        except Exception as e:
            self.logger.error(f"Error fetching data: {str(e)}")
            return pd.DataFrame(columns=['date', 'Category', 'Variable', 'Value', 'Frequency'])

    def aggregate_to_monthly(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate data to monthly frequency based on each series' original frequency
        """
        # Check if the dataframe is empty
        if df.empty:
            self.logger.warning("Input DataFrame is empty, returning empty monthly DataFrame")
            return pd.DataFrame(columns=['date', 'Category', 'Variable', 'Value'])
            
        df['date'] = pd.to_datetime(df['date'])
        df['YearMonth'] = df['date'].dt.to_period('M')
        
        # Handle special null cases
        df['Frequency'] = df['Frequency'].fillna('Weekly')

        dfs = []
        
        # Daily and Weekly data: average by month
        daily_weekly = df[df['Frequency'].isin(['Intraday','Daily', 'Weekly','Weekly on Friday','Weekly on Monday'])]
        if not daily_weekly.empty:
            monthly_avg = daily_weekly.groupby(['YearMonth', 'Category', 'Variable'])['Value'].mean().reset_index()
            dfs.append(monthly_avg)
            self.logger.info(f"Processed {len(daily_weekly)} rows of daily/weekly data")
        
        # Quarterly data: expand to months
        quarterly = df[df['Frequency'] == 'Quarterly']
        if not quarterly.empty:
            quarterly_expanded = []
            for _, row in quarterly.iterrows():
                quarter_start = row['YearMonth'].to_timestamp()
                for month in range(3):
                    monthly_expanded = row.copy()
                    monthly_expanded['YearMonth'] = pd.Period(quarter_start + pd.DateOffset(months=month), freq='M')
                    quarterly_expanded.append(monthly_expanded)
            if quarterly_expanded:
                quarterly_df = pd.DataFrame(quarterly_expanded)
                dfs.append(quarterly_df)
                self.logger.info(f"Processed {len(quarterly)} rows of quarterly data")
        
        # Monthly data: keep as is
        monthly = df[df['Frequency'] == 'Monthly']
        if not monthly.empty:
            dfs.append(monthly)
            self.logger.info(f"Processed {len(monthly)} rows of monthly data")
        
        # Check if dfs is empty
        if not dfs:
            self.logger.warning("No data frames to concatenate, returning empty DataFrame")
            return pd.DataFrame(columns=['date', 'Category', 'Variable', 'Value'])
            
        try:
            result = pd.concat(dfs, ignore_index=True)
            result['date'] = result['YearMonth'].dt.to_timestamp()
            
            return result[['date', 'Category', 'Variable', 'Value']].sort_values(['date', 'Category', 'Variable'])
        except Exception as e:
            self.logger.error(f"Error in aggregation: {str(e)}")
            return pd.DataFrame(columns=['date', 'Category', 'Variable', 'Value'])


class EIADataFetcher:
    def __init__(self, synapse_connector):
        self.connector = synapse_connector
        self.logger = logging.getLogger(__name__)
        
    def fetch_fuel_demand_data(self) -> pd.DataFrame:
        """
        Fetch US diesel and gasoline demand data from EIA
        - Diesel: PET.MDIUPUS2.M
        - Gasoline: PET.MGFUPUS2.M
        
        Returns:
            pd.DataFrame: DataFrame with date, diesel and gasoline demand
        """
        query = """
        SELECT 
            date,
            SUM(CASE WHEN series_id = 'PET.MDIUPUS2.M' THEN value ELSE 0 END) as Diesel,
            SUM(CASE WHEN series_id = 'PET.MGFUPUS2.M' THEN value ELSE 0 END) as Gasoline
        FROM [atlas].[standard_eiapetroleum_timeseries_v3latest]
        WHERE series_id IN ('PET.MDIUPUS2.M', 'PET.MGFUPUS2.M')
        GROUP BY date
        ORDER BY date ASC;
        """
        
        # Add retry logic
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                # Force reconnection before each attempt to avoid "closed automatically" error
                if retry_count > 0:
                    self.logger.info(f"Reconnecting to database for retry attempt {retry_count+1}")
                    self.connector.connect()
                    time.sleep(2)  # Add a small delay
                
                df = self.connector.query_to_df(query)
                if df is not None and not df.empty:
                    self.logger.info(f"Retrieved {len(df)} rows of US fuel demand data")
                    df['date'] = pd.to_datetime(df['date'])
                    return df
                else:
                    self.logger.warning(f"Empty result on attempt {retry_count+1}. Retrying...")
                    retry_count += 1
                    time.sleep(3)  # Longer delay for empty results
                    
            except Exception as e:
                error_msg = str(e)
                self.logger.warning(f"Error on attempt {retry_count+1}: {error_msg}")
                
                if "closed automatically" in error_msg:
                    self.logger.info("Connection was closed. Will reconnect and retry.")
                    retry_count += 1
                    time.sleep(3)
                else:
                    # For other errors, log and return empty DataFrame
                    self.logger.error(f"Unexpected error fetching fuel demand data: {error_msg}")
                    return pd.DataFrame(columns=['date', 'Diesel', 'Gasoline'])
        
        self.logger.error(f"Failed to retrieve EIA data after {max_retries} attempts")
        return pd.DataFrame(columns=['date', 'Diesel', 'Gasoline'])


class DataProcessor:
    @staticmethod
    def calculate_percent_change(df):
        """
        Calculate percent changes for all numeric columns in a dataframe
        """
        logger = logging.getLogger(__name__)
        
        if df.empty:
            logger.warning("Cannot calculate percent change on empty DataFrame")
            return df
            
        try:
            return df.pct_change() * 100
        except Exception as e:
            logger.error(f"Error calculating percent change: {str(e)}")
            return df
    
    @staticmethod
    def prepare_data(input_df, demand_df):
        """
        Prepare data for analysis, including percent change calculation
        
        Args:
            input_df (pd.DataFrame): Input features data from Bloomberg
            demand_df (pd.DataFrame): Target data for fuel demand
            
        Returns:
            tuple: Processed dataframes and category mapping
        """
        logger = logging.getLogger(__name__)
        
        # Check for empty dataframes with better error messages
        if input_df.empty:
            logger.error("Input features DataFrame is empty. Cannot proceed with data preparation.")
            empty_result = (
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
            )
            return empty_result
            
        if demand_df.empty:
            logger.error("Demand DataFrame is empty. Cannot proceed with data preparation.")
            
            # In this case, we'll create dummy structures but only with input data
            # This allows us to still work with the input features even if demand data is missing
            
            # Get category info from input data
            category_map = input_df.set_index('Variable')['Category'].to_dict()
            
            # Create pivot table for input data
            input_df['date'] = pd.to_datetime(input_df['date'])
            input_pivot = input_df.pivot(index='date', columns='Variable', values='Value')
            
            # Calculate percent changes for input data
            input_pivot_pct = DataProcessor.calculate_percent_change(input_pivot)
            
            # Return input data with empty demand data
            empty_demand = pd.DataFrame(index=input_pivot.index)
            empty_demand['Diesel'] = np.nan
            empty_demand['Gasoline'] = np.nan
            
            empty_demand_pct = pd.DataFrame(index=input_pivot_pct.index)
            empty_demand_pct['Diesel'] = np.nan
            empty_demand_pct['Gasoline'] = np.nan
            
            logger.warning("Proceeding with empty demand data. Only input features will be valid.")
            
            return input_pivot, input_pivot_pct, empty_demand, empty_demand_pct, category_map
        
        try:
            # Convert dates to datetime
            input_df['date'] = pd.to_datetime(input_df['date'])
            demand_df['date'] = pd.to_datetime(demand_df['date'])
            
            # Set date as index for demand_df
            demand_df = demand_df.set_index('date')
            
            # Create pivot table for input data
            input_pivot = input_df.pivot(index='date', columns='Variable', values='Value')
            
            # Calculate percent changes
            input_pivot_pct = DataProcessor.calculate_percent_change(input_pivot)
            demand_df_pct = DataProcessor.calculate_percent_change(demand_df)
            
            # Add category information
            category_map = input_df.set_index('Variable')['Category'].to_dict()
            
            # Align dates between the two dataframes
            start_date = max(input_pivot.index.min(), demand_df.index.min())
            end_date = min(input_pivot.index.max(), demand_df.index.max())
            
            logger.info(f"Data alignment: start_date = {start_date}, end_date = {end_date}")
            
            input_pivot = input_pivot.loc[start_date:end_date]
            input_pivot_pct = input_pivot_pct.loc[start_date:end_date]
            demand_df_absolute = demand_df.loc[start_date:end_date]
            demand_df_pct = demand_df_pct.loc[start_date:end_date]
            
            # Ensure index names match
            input_pivot_pct.index.name = 'date'
            
            # Remove the first row which will be NaN due to percent change calculation
            input_pivot_pct = input_pivot_pct.dropna(how='all')
            demand_df_pct = demand_df_pct.dropna(how='all')
            
            logger.info(f"Processed data: {len(input_pivot)} rows of input data, {len(demand_df_absolute)} rows of demand data")
            
            return input_pivot, input_pivot_pct, demand_df_absolute, demand_df_pct, category_map
            
        except Exception as e:
            logger.error(f"Error in data preparation: {str(e)}", exc_info=True)
            empty_result = (
                pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}
            )
            return empty_result