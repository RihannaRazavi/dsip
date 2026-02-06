"""
Statistical summary functions for redwood data analysis
"""
import pandas as pd
import numpy as np


def calculate_gradient_statistics(data):
    """
    Calculate vertical gradient statistics for all environmental variables
    
    Parameters
    ----------
    data : pd.DataFrame
        Analysis-ready dataframe with Height and environmental variables
    
    Returns
    -------
    dict
        Dictionary containing gradient statistics
    """
    stats = {}
    
    # Temperature statistics
    temp_by_height = data.groupby('Height')['humid_temp'].agg(['mean', 'min', 'max', 'std'])
    stats['temp_range'] = temp_by_height['mean'].max() - temp_by_height['mean'].min()
    stats['temp_top_mean'] = temp_by_height['mean'].max()
    stats['temp_bottom_mean'] = temp_by_height['mean'].min()
    stats['temp_top_height'] = temp_by_height['mean'].idxmax()
    stats['temp_bottom_height'] = temp_by_height['mean'].idxmin()
    
    # Humidity statistics
    humid_by_height = data.groupby('Height')['humidity'].agg(['mean', 'min', 'max', 'std'])
    stats['humid_range'] = humid_by_height['mean'].max() - humid_by_height['mean'].min()
    stats['humid_top_mean'] = humid_by_height['mean'].min()  # Top is drier
    stats['humid_bottom_mean'] = humid_by_height['mean'].max()  # Bottom is more humid
    
    # Light statistics (daytime only)
    daytime = data[(data['hour'] >= 10) & (data['hour'] <= 16) & (data['hamatop'] > 0)]
    if len(daytime) > 0:
        par_by_height = daytime.groupby('Height')['hamatop'].agg(['mean', 'min', 'max', 'std'])
        if len(par_by_height) > 0:
            stats['light_top_mean'] = par_by_height['mean'].max()
            stats['light_bottom_mean'] = par_by_height['mean'].min()
            stats['light_reduction_pct'] = 100 * (1 - stats['light_bottom_mean'] / stats['light_top_mean'])
    
    return stats


def summarize_by_layer(data):
    """
    Calculate summary statistics by canopy layer
    
    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with canopy_layer column
    
    Returns
    -------
    pd.DataFrame
        Summary statistics by layer
    """
    summary = data.groupby('canopy_layer').agg({
        'humid_temp': ['mean', 'std', 'min', 'max'],
        'humidity': ['mean', 'std', 'min', 'max'],
        'hamatop': ['mean', 'std', 'min', 'max'],
        'Height': ['min', 'max', 'count']
    }).round(2)
    
    return summary


def calculate_data_quality_metrics(original_data, cleaned_data):
    """
    Calculate data quality metrics after cleaning
    
    Parameters
    ----------
    original_data : pd.DataFrame
        Original uncleaned data
    cleaned_data : pd.DataFrame
        Cleaned data
    
    Returns
    -------
    dict
        Data quality metrics
    """
    metrics = {}
    
    metrics['original_rows'] = len(original_data)
    metrics['cleaned_rows'] = len(cleaned_data)
    metrics['removed_rows'] = metrics['original_rows'] - metrics['cleaned_rows']
    metrics['retention_pct'] = 100 * metrics['cleaned_rows'] / metrics['original_rows']
    
    # Missing data by variable
    metrics['missing_original'] = original_data.isnull().sum().to_dict()
    metrics['missing_cleaned'] = cleaned_data.isnull().sum().to_dict()
    
    return metrics