"""
Data preparation functions for redwood analysis
"""
import pandas as pd
import numpy as np

def merge_with_locations(redwood_df, motes_df):
    
    # Ensure ID column is same type as nodeid for merging
    motes_df = motes_df.copy()
    motes_df['ID'] = motes_df['ID'].astype(int)
    
    merged = redwood_df.merge(
        motes_df, 
        left_on='nodeid', 
        right_on='ID', 
        how='left'
    )
    return merged


def add_time_features(data):
    
    data = data.copy()
    
    # Ensure datetime type
    data['result_time'] = pd.to_datetime(data['result_time'])
    
    # Extract time features
    data['hour'] = data['result_time'].dt.hour
    data['date'] = data['result_time'].dt.date
    data['day_of_week'] = data['result_time'].dt.day_name()
    
    return data


def classify_canopy_layers(data, lower_bound=35, upper_bound=55):
    data = data.copy()
    
    data['canopy_layer'] = pd.cut(
        data['Height'], 
        bins=[0, lower_bound, upper_bound, 100],
        labels=['Lower', 'Middle', 'Upper']
    )
    
    return data


def prepare_analysis_data(redwood_df, motes_df):
    
    # Merge with locations
    data = merge_with_locations(redwood_df, motes_df)
    
    # Add time features
    data = add_time_features(data)
    
    # Classify canopy layers
    data = classify_canopy_layers(data)
    
    return data


def filter_daytime_data(data, start_hour=10, end_hour=16):
    
    daytime = data[
        (data['hour'] >= start_hour) & 
        (data['hour'] <= end_hour)
    ].copy()
    
    return daytime
