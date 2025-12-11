"""
Data loading, preprocessing, and demand forecasting module.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Optional


def load_sales_data(file_path) -> pd.DataFrame:
    """
    Load sales history CSV file.
    
    Args:
        file_path: Path to sales history CSV file (str, Path, or file-like object)
        
    Returns:
        DataFrame with columns: date, sku, demand
    """
    df = pd.read_csv(file_path)
    df['date'] = pd.to_datetime(df['date'])
    return df


def load_sku_data(file_path) -> pd.DataFrame:
    """
    Load SKU parameters CSV file.
    
    Args:
        file_path: Path to SKU costs CSV file (str, Path, or file-like object)
        
    Returns:
        DataFrame with columns: sku, unit_cost, holding_cost, stockout_penalty, volume, fixed_order_cost
    """
    df = pd.read_csv(file_path)
    # Fill missing fixed_order_cost with 0
    if 'fixed_order_cost' not in df.columns:
        df['fixed_order_cost'] = 0.0
    else:
        df['fixed_order_cost'] = df['fixed_order_cost'].fillna(0.0)
    return df


def aggregate_weekly(sales_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily sales data to weekly totals.
    
    Args:
        sales_df: DataFrame with date, sku, demand columns
        
    Returns:
        DataFrame aggregated by week and SKU
    """
    # Ensure date is datetime
    sales_df = sales_df.copy()
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    
    # Create week identifier
    sales_df['year_week'] = sales_df['date'].dt.to_period('W')
    
    # Aggregate by week and SKU
    weekly_df = sales_df.groupby(['year_week', 'sku'])['demand'].sum().reset_index()
    weekly_df = weekly_df.rename(columns={'year_week': 'week'})
    
    return weekly_df


def forecast_demand_moving_average(
    sales_df: pd.DataFrame,
    forecast_horizon: int = 4,
    window_size: Optional[int] = None
) -> Dict[str, float]:
    """
    Generate demand forecasts using simple moving average.
    
    Args:
        sales_df: DataFrame with date, sku, demand columns
        forecast_horizon: Number of weeks to forecast (default: 4)
        window_size: Number of weeks to use for moving average (default: all available)
        
    Returns:
        Dictionary mapping SKU to forecasted demand (total over horizon)
    """
    # Aggregate to weekly
    weekly_df = aggregate_weekly(sales_df)
    
    # Calculate moving average for each SKU
    forecasts = {}
    
    for sku in weekly_df['sku'].unique():
        sku_data = weekly_df[weekly_df['sku'] == sku]['demand'].values
        
        if len(sku_data) == 0:
            forecasts[sku] = 0.0
            continue
        
        # Use all available data if window_size not specified
        if window_size is None or window_size >= len(sku_data):
            avg_weekly_demand = np.mean(sku_data)
        else:
            avg_weekly_demand = np.mean(sku_data[-window_size:])
        
        # Forecast = average weekly demand * forecast horizon
        forecasts[sku] = avg_weekly_demand * forecast_horizon
    
    return forecasts


def prepare_optimization_data(
    sales_file,
    sku_file,
    forecast_horizon: int = 4
) -> Tuple[Dict[str, float], pd.DataFrame]:
    """
    Prepare all data needed for optimization.
    
    Args:
        sales_file: Path to sales history CSV
        sku_file: Path to SKU costs CSV
        forecast_horizon: Number of weeks to forecast
        
    Returns:
        Tuple of (demand_forecasts dict, sku_params DataFrame)
    """
    # Load data
    sales_df = load_sales_data(sales_file)
    sku_df = load_sku_data(sku_file)
    
    # Generate forecasts
    forecasts = forecast_demand_moving_average(sales_df, forecast_horizon)
    
    # Merge forecasts into SKU dataframe
    sku_df['demand_forecast'] = sku_df['sku'].map(forecasts).fillna(0.0)
    
    return forecasts, sku_df

