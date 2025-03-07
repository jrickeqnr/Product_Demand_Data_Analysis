import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import logging
import argparse
from config_loader import load_config

def setup_logging(config):
    """Set up logging based on configuration"""
    log_level = getattr(logging, config['logging']['level'])
    handlers = []
    
    if config['logging']['file_logging']:
        handlers.append(logging.FileHandler("eda.log"))
    
    if config['logging']['console_logging']:
        handlers.append(logging.StreamHandler())
    
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    
    return logging.getLogger(__name__)

class FuelDemandAnalyzer:
    def __init__(self, config):
        self.config = config
        data_dir = config['paths']['data_dir']
        charts_dir = config['paths']['charts_dir']
        
        # Create output directory for charts and visualizations
        os.makedirs(charts_dir, exist_ok=True)
        self.charts_dir = charts_dir
        
        # Load data
        self.input_absolute = pd.read_csv(f'{data_dir}/input_features_absolute.csv', parse_dates=['Date'], index_col='Date')
        self.input_pct = pd.read_csv(f'{data_dir}/input_features_percent_change.csv', parse_dates=['date'], index_col='date')
        self.demand_absolute = pd.read_csv(f'{data_dir}/demand_absolute.csv', parse_dates=['date'], index_col='date')
        self.demand_pct = pd.read_csv(f'{data_dir}/demand_percent_change.csv', parse_dates=['date'], index_col='date')
        
        # Load category mapping
        with open(f'{data_dir}/category_map.pkl', 'rb') as f:
            self.category_map = pickle.load(f)
            
        logger.info(f"Loaded data with {len(self.input_absolute)} rows and {self.input_absolute.shape[1]} features")
        
        # Configure visualization settings
        self.dpi = config['visualization']['dpi']
        self.save_formats = config['visualization']['save_formats']
        self.charts_to_generate = config['visualization']['charts_to_generate']
        
    def save_figure(self, filename_base):
        """Save figure in all configured formats"""
        for fmt in self.save_formats:
            full_path = os.path.join(self.charts_dir, f"{filename_base}.{fmt}")
            plt.savefig(full_path, dpi=self.dpi)
        plt.close()
        
    def analyze_data_structure(self):
        """
        Analyze and print information about the data structure
        """
        logger.info("Analyzing data structure...")
        
        # Print time ranges
        logger.info(f"Data time range: {self.input_absolute.index.min()} to {self.input_absolute.index.max()}")
        
        # Print column categories
        category_counts = {}
        for variable, category in self.category_map.items():
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1
                
        logger.info("\nFeatures by category:")
        for category, count in category_counts.items():
            logger.info(f"  {category}: {count} variables")
            
        # Print information about the demand data
        logger.info("\nFuel demand data:")
        logger.info(self.demand_absolute.describe())
        
        # Check for missing values
        missing_values = self.input_absolute.isna().sum()
        logger.info("\nFeatures with missing values:")
        logger.info(missing_values[missing_values > 0])
        
    def plot_demand_trends(self):
        """
        Plot historical trends in gasoline and diesel demand
        """
        if 'trends' not in self.charts_to_generate:
            logger.info("Skipping demand trends charts based on configuration")
            return
            
        logger.info("Plotting demand trends...")
        
        # Create figure
        plt.figure(figsize=(12, 8))
        
        # Plot diesel demand
        plt.subplot(2, 1, 1)
        self.demand_absolute['Diesel'].plot()
        plt.title('US Diesel Demand (Million Barrels per Day)')
        plt.grid(True)
        
        # Plot gasoline demand
        plt.subplot(2, 1, 2)
        self.demand_absolute['Gasoline'].plot()
        plt.title('US Gasoline Demand (Million Barrels per Day)')
        plt.grid(True)
        
        plt.tight_layout()
        self.save_figure('demand_trends')
        
        # Plot year-over-year growth rates
        plt.figure(figsize=(12, 8))
        
        # Calculate YoY growth rates
        yoy_growth = self.demand_absolute.pct_change(periods=12) * 100
        
        # Plot diesel YoY growth
        plt.subplot(2, 1, 1)
        yoy_growth['Diesel'].plot()
        plt.title('US Diesel Demand YoY Growth (%)')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.grid(True)
        
        # Plot gasoline YoY growth
        plt.subplot(2, 1, 2)
        yoy_growth['Gasoline'].plot()
        plt.title('US Gasoline Demand YoY Growth (%)')
        plt.axhline(y=0, color='r', linestyle='-')
        plt.grid(True)
        
        plt.tight_layout()
        self.save_figure('demand_yoy_growth')
        
        # Plot seasonality if enabled
        if 'seasonality' in self.charts_to_generate:
            plt.figure(figsize=(12, 8))
            
            # Create month column
            demand_monthly = self.demand_absolute.copy()
            demand_monthly['month'] = demand_monthly.index.month
            
            # Plot diesel seasonality
            plt.subplot(2, 1, 1)
            sns.boxplot(x='month', y='Diesel', data=demand_monthly.reset_index())
            plt.title('US Diesel Demand Seasonality')
            plt.grid(True)
            
            # Plot gasoline seasonality
            plt.subplot(2, 1, 2)
            sns.boxplot(x='month', y='Gasoline', data=demand_monthly.reset_index())
            plt.title('US Gasoline Demand Seasonality')
            plt.grid(True)
            
            plt.tight_layout()
            self.save_figure('demand_seasonality')
        
    def analyze_correlations(self):
        """
        Analyze correlations between features and demand
        """
        if 'correlations' not in self.charts_to_generate:
            logger.info("Skipping correlation analysis based on configuration")
            return
            
        logger.info("Analyzing correlations...")
        
        # Calculate correlations with demand (both absolute and percent change)
        abs_corr_diesel = self.input_absolute.corrwith(self.demand_absolute['Diesel']).sort_values(ascending=False)
        abs_corr_gasoline = self.input_absolute.corrwith(self.demand_absolute['Gasoline']).sort_values(ascending=False)
        
        pct_corr_diesel = self.input_pct.corrwith(self.demand_pct['Diesel']).sort_values(ascending=False)
        pct_corr_gasoline = self.input_pct.corrwith(self.demand_pct['Gasoline']).sort_values(ascending=False)
        
        # Print top correlations
        logger.info("\nTop 10 features correlated with Diesel demand (absolute values):")
        logger.info(abs_corr_diesel.head(10))
        logger.info("\nTop 10 features correlated with Gasoline demand (absolute values):")
        logger.info(abs_corr_gasoline.head(10))
        
        logger.info("\nTop 10 features correlated with Diesel demand changes:")
        logger.info(pct_corr_diesel.head(10))
        logger.info("\nTop 10 features correlated with Gasoline demand changes:")
        logger.info(pct_corr_gasoline.head(10))
        
        # Save correlations to CSV for reference
        abs_corr_diesel.to_csv(os.path.join(self.charts_dir, 'diesel_absolute_correlations.csv'))
        abs_corr_gasoline.to_csv(os.path.join(self.charts_dir, 'gasoline_absolute_correlations.csv'))
        pct_corr_diesel.to_csv(os.path.join(self.charts_dir, 'diesel_pct_change_correlations.csv'))
        pct_corr_gasoline.to_csv(os.path.join(self.charts_dir, 'gasoline_pct_change_correlations.csv'))
        
        # Plot top correlations
        plt.figure(figsize=(14, 10))
        
        # Add category information to correlation data
        abs_corr_diesel_df = abs_corr_diesel.reset_index()
        abs_corr_diesel_df.columns = ['Variable', 'Correlation']
        abs_corr_diesel_df['Category'] = abs_corr_diesel_df['Variable'].map(self.category_map)
        
        # Plot top positive and negative correlations
        top_positive = abs_corr_diesel_df.nlargest(10, 'Correlation')
        top_negative = abs_corr_diesel_df.nsmallest(10, 'Correlation')
        
        plt.subplot(2, 1, 1)
        sns.barplot(x='Correlation', y='Variable', hue='Category', data=top_positive)
        plt.title('Top Positive Correlations with Diesel Demand')
        plt.tight_layout()
        
        plt.subplot(2, 1, 2)
        sns.barplot(x='Correlation', y='Variable', hue='Category', data=top_negative)
        plt.title('Top Negative Correlations with Diesel Demand')
        
        plt.tight_layout()
        self.save_figure('diesel_correlations')
        
        # Repeat for gasoline
        abs_corr_gasoline_df = abs_corr_gasoline.reset_index()
        abs_corr_gasoline_df.columns = ['Variable', 'Correlation']
        abs_corr_gasoline_df['Category'] = abs_corr_gasoline_df['Variable'].map(self.category_map)
        
        top_positive = abs_corr_gasoline_df.nlargest(10, 'Correlation')
        top_negative = abs_corr_gasoline_df.nsmallest(10, 'Correlation')
        
        plt.figure(figsize=(14, 10))
        
        plt.subplot(2, 1, 1)
        sns.barplot(x='Correlation', y='Variable', hue='Category', data=top_positive)
        plt.title('Top Positive Correlations with Gasoline Demand')
        
        plt.subplot(2, 1, 2)
        sns.barplot(x='Correlation', y='Variable', hue='Category', data=top_negative)
        plt.title('Top Negative Correlations with Gasoline Demand')
        
        plt.tight_layout()
        self.save_figure('gasoline_correlations')
        
        # Create correlation matrix heatmap
        # Select top 20 correlated features for each fuel type
        top_diesel_vars = list(abs_corr_diesel.index[:15]) + list(abs_corr_diesel.index[-5:])
        top_gasoline_vars = list(abs_corr_gasoline.index[:15]) + list(abs_corr_gasoline.index[-5:])
        
        # Combine unique variables
        top_vars = list(set(top_diesel_vars + top_gasoline_vars))
        
        # Create correlation matrix
        combined_data = pd.concat([self.input_absolute[top_vars], self.demand_absolute], axis=1)
        corr_matrix = combined_data.corr()
        
        # Plot heatmap
        plt.figure(figsize=(16, 14))
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', 
                    vmin=-1, vmax=1, center=0, linewidths=.5)
        plt.title('Correlation Matrix: Top Features vs Fuel Demand')
        plt.tight_layout()
        self.save_figure('correlation_matrix')
        
    def perform_pca_analysis(self):
        """
        Perform PCA analysis to understand feature relationships
        """
        if 'pca' not in self.charts_to_generate or not self.config['analysis']['pca']['enabled']:
            logger.info("Skipping PCA analysis based on configuration")
            return
            
        logger.info("Performing PCA analysis...")
        
        # Fill missing values with median for PCA
        input_filled = self.input_absolute.fillna(self.input_absolute.median())
        
        # Standardize the data
        scaler = StandardScaler()
        input_scaled = scaler.fit_transform(input_filled)
        
        # Perform PCA
        pca = PCA()
        pca_result = pca.fit_transform(input_scaled)
        
        # Plot explained variance
        plt.figure(figsize=(10, 6))
        cum_explained_variance = np.cumsum(pca.explained_variance_ratio_)
        plt.plot(cum_explained_variance)
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.grid(True)
        plt.axhline(y=0.8, color='r', linestyle='--')
        plt.axhline(y=0.9, color='g', linestyle='--')
        plt.title('PCA Explained Variance')
        self.save_figure('pca_explained_variance')
        
        # Get feature loadings for top components
        n_components = 3  # Number of components to analyze
        loadings = pd.DataFrame(
            pca.components_.T[:, :n_components], 
            columns=[f'PC{i+1}' for i in range(n_components)],
            index=input_filled.columns
        )
        
        # Add category information
        loadings['Category'] = loadings.index.map(self.category_map)
        
        # Plot feature loadings
        plt.figure(figsize=(15, 10))
        for i in range(n_components):
            plt.subplot(n_components, 1, i+1)
            top_features = loadings.iloc[:, i].abs().sort_values(ascending=False).head(10).index
            sns.barplot(x=loadings.loc[top_features, f'PC{i+1}'], y=top_features)
            plt.title(f'Top Features in Principal Component {i+1}')
            plt.tight_layout()
        
        self.save_figure('pca_feature_loadings')
        
        # Plot first two components with category colors
        plt.figure(figsize=(12, 10))
        loadings_plot = loadings.reset_index()
        
        # Create a scatter plot of PC1 vs PC2
        sns.scatterplot(x='PC1', y='PC2', hue='Category', data=loadings_plot, s=100, alpha=0.7)
        
        # Add labels for important features
        top_features = loadings.iloc[:, :2].apply(lambda x: x.abs().sum(), axis=1).sort_values(ascending=False).head(15).index
        for feature in top_features:
            plt.annotate(feature, 
                         xy=(loadings.loc[feature, 'PC1'], loadings.loc[feature, 'PC2']),
                         xytext=(5, 5),
                         textcoords='offset points')
        
        plt.title('Feature Distribution in First Two Principal Components')
        plt.grid(True)
        plt.tight_layout()
        self.save_figure('pca_feature_distribution')
        
    def rolling_regression_analysis(self, window_size=36, target='Diesel'):
        """
        Perform rolling regression analysis to see how relationships change over time
        
        Args:
            window_size (int): Size of the rolling window in months
            target (str): Target variable ('Diesel' or 'Gasoline')
        """
        logger.info(f"Performing rolling regression analysis for {target} with window size {window_size}...")
        
        # Prepare data
        X = self.input_absolute.copy()
        y = self.demand_absolute[target].copy()
        
        # Fill missing values with median
        X = X.fillna(X.median())
        
        # Ensure alignment
        common_idx = X.index.intersection(y.index)
        X = X.loc[common_idx]
        y = y.loc[common_idx]
        
        # Initialize results storage
        dates = []
        r2_scores = []
        top_features = {}
        
        # Run rolling regression
        for start_idx in range(0, len(X) - window_size + 1):
            end_idx = start_idx + window_size
            
            # Get data for this window
            X_window = X.iloc[start_idx:end_idx]
            y_window = y.iloc[start_idx:end_idx]
            
            # Fit regression model
            model = LinearRegression()
            model.fit(X_window, y_window)
            
            # Store results
            window_end_date = X.index[end_idx - 1]
            dates.append(window_end_date)
            
            # Calculate R^2
            y_pred = model.predict(X_window)
            r2 = r2_score(y_window, y_pred)
            r2_scores.append(r2)
            
            # Get feature importance
            importance = pd.Series(model.coef_, index=X.columns)
            
            # Store top positive and negative features
            top_pos = importance.nlargest(5).index.tolist()
            top_neg = importance.nsmallest(5).index.tolist()
            top_features[window_end_date] = {'positive': top_pos, 'negative': top_neg}
        
        # Plot R^2 over time
        plt.figure(figsize=(12, 6))
        plt.plot(dates, r2_scores)
        plt.title(f'Rolling Regression R² for {target} Demand (Window Size: {window_size} months)')
        plt.xlabel('Window End Date')
        plt.ylabel('R²')
        plt.grid(True)
        plt.tight_layout()
        self.save_figure(f'{target.lower()}_rolling_r2')
        
        # Analyze feature stability
        feature_counts = {}
        for date_features in top_features.values():
            for feature in date_features['positive'] + date_features['negative']:
                if feature in feature_counts:
                    feature_counts[feature] += 1
                else:
                    feature_counts[feature] = 1
        
        # Get most stable features
        stable_features = pd.Series(feature_counts).sort_values(ascending=False).head(10)
        
        plt.figure(figsize=(12, 6))
        stable_features.plot(kind='bar')
        plt.title(f'Most Stable Important Features for {target} Demand')
        plt.xlabel('Feature')
        plt.ylabel('Frequency in Top Features')
        plt.grid(True)
        plt.tight_layout()
        self.save_figure(f'{target.lower()}_stable_features')
        
        return dates, r2_scores, top_features
        
    def cross_correlation_analysis(self, target='Diesel', max_lag=12):
        """
        Perform cross-correlation analysis to identify leading indicators
        
        Args:
            target (str): Target variable ('Diesel' or 'Gasoline')
            max_lag (int): Maximum lag in months to test
            
        Returns:
            pd.DataFrame: DataFrame with cross-correlation results
        """
        logger.info(f"Performing cross-correlation analysis for {target} with max lag {max_lag}...")
        
        # Get target series
        target_series = self.demand_pct[target]
        
        # Initialize results
        results = []
        
        # Calculate cross-correlations for each feature
        for column in self.input_pct.columns:
            feature_series = self.input_pct[column]
            
            # Calculate cross-correlation
            cross_corr = [
                stats.pearsonr(
                    feature_series.shift(lag).dropna(),
                    target_series.loc[feature_series.shift(lag).dropna().index]
                )[0]
                for lag in range(max_lag + 1)
            ]
            
            # Find max correlation and corresponding lag
            max_corr = max(cross_corr, key=abs)
            optimal_lag = cross_corr.index(max_corr)
            
            # Store results
            category = self.category_map.get(column, 'Unknown')
            results.append({
                'Variable': column,
                'Category': category,
                'Optimal_Lag': optimal_lag,
                'Max_Correlation': max_corr
            })
        
        # Convert to DataFrame
        results_df = pd.DataFrame(results)
        
        # Sort by absolute correlation
        results_df['Abs_Correlation'] = results_df['Max_Correlation'].abs()
        results_df = results_df.sort_values('Abs_Correlation', ascending=False).reset_index(drop=True)
        
        # Plot top leading indicators
        top_indicators = results_df.head(15)
        
        plt.figure(figsize=(12, 8))
        bars = plt.barh(top_indicators['Variable'], top_indicators['Max_Correlation'], color='skyblue')
        
        # Add lag information to bars
        for i, bar in enumerate(bars):
            lag = top_indicators.iloc[i]['Optimal_Lag']
            if lag > 0:
                plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                        f'Lag: {lag}m', va='center')
        
        plt.title(f'Top Leading Indicators for {target} Demand')
        plt.xlabel('Maximum Cross-Correlation')
        plt.tight_layout()
        self.save_figure(f'{target.lower()}_leading_indicators')
        
        # Save results to CSV
        results_df.to_csv(os.path.join(self.charts_dir, f'{target.lower()}_cross_correlations.csv'), index=False)
        
        return results_df

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Perform exploratory data analysis on fuel demand data')
    parser.add_argument('--config', default='config.json', help='Path to configuration file')
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Set up logging
    global logger
    logger = setup_logging(config)
    
    try:
        logger.info("Starting exploratory data analysis...")
        
        # Initialize analyzer
        analyzer = FuelDemandAnalyzer(config)
        
        # Analyze data structure
        analyzer.analyze_data_structure()
        
        # Plot demand trends
        analyzer.plot_demand_trends()
        
        # Analyze correlations
        analyzer.analyze_correlations()
        
        # Perform PCA analysis
        analyzer.perform_pca_analysis()
        
        # Perform rolling regression analysis if targets are enabled
        if config['targets']['diesel']:
            analyzer.rolling_regression_analysis(window_size=36, target='Diesel')
        
        if config['targets']['gasoline']:
            analyzer.rolling_regression_analysis(window_size=36, target='Gasoline')
        
        # Perform cross-correlation analysis
        if config['targets']['diesel']:
            diesel_indicators = analyzer.cross_correlation_analysis(target='Diesel', max_lag=12)
        
        if config['targets']['gasoline']:
            gasoline_indicators = analyzer.cross_correlation_analysis(target='Gasoline', max_lag=12)
        
        logger.info("Exploratory data analysis completed successfully.")
        
    except Exception as e:
        logger.error(f"Error in exploratory data analysis: {str(e)}", exc_info=True)

if __name__ == "__main__":
    main()