"""
Visualization functions for redwood microclimate analysis
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def plot_vertical_gradients(data, figsize=(16, 6)):
    """
    Create three-panel plot showing temperature, humidity, and light gradients
    
    Parameters
    ----------
    data : pd.DataFrame
        Analysis-ready dataframe
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
    
    # Temperature gradient
    temp_stats = data.groupby('Height')['humid_temp'].agg(['mean', 'std', 'count'])
    temp_stats = temp_stats[temp_stats['count'] > 100]
    
    ax1.errorbar(temp_stats['mean'], temp_stats.index, 
                 xerr=temp_stats['std'], fmt='o', capsize=5,
                 color='orangered', markersize=10, linewidth=2, capthick=2,
                 markeredgecolor='darkred', markeredgewidth=1.5)
    ax1.set_xlabel('Temperature (°C)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Height Above Ground (m)', fontsize=13, fontweight='bold')
    ax1.set_title('Temperature Gradient', fontsize=15, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # Humidity gradient
    humid_stats = data.groupby('Height')['humidity'].agg(['mean', 'std', 'count'])
    humid_stats = humid_stats[humid_stats['count'] > 100]
    
    ax2.errorbar(humid_stats['mean'], humid_stats.index,
                 xerr=humid_stats['std'], fmt='o', capsize=5,
                 color='dodgerblue', markersize=10, linewidth=2, capthick=2,
                 markeredgecolor='darkblue', markeredgewidth=1.5)
    ax2.set_xlabel('Relative Humidity (%)', fontsize=13, fontweight='bold')
    ax2.set_ylabel('Height Above Ground (m)', fontsize=13, fontweight='bold')
    ax2.set_title('Humidity Gradient', fontsize=15, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    # Light gradient (daytime only)
    daytime = data[(data['hour'] >= 10) & (data['hour'] <= 16) & (data['hamatop'] > 100)]
    light_stats = daytime.groupby('Height')['hamatop'].agg(['mean', 'std', 'count'])
    light_stats = light_stats[light_stats['count'] > 50]
    
    ax3.errorbar(light_stats['mean'], light_stats.index,
                 xerr=light_stats['std'], fmt='o', capsize=5,
                 color='gold', markersize=10, linewidth=2, capthick=2,
                 markeredgecolor='darkgoldenrod', markeredgewidth=1.5)
    ax3.set_xlabel('Incident PAR (µmol/m²/s)', fontsize=13, fontweight='bold')
    ax3.set_ylabel('Height Above Ground (m)', fontsize=13, fontweight='bold')
    ax3.set_title('Light Gradient', fontsize=15, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig


def plot_temp_humidity_by_layer(data, figsize=(16, 5)):
    """
    Plot temperature vs humidity relationship for different canopy layers
    
    Parameters
    ----------
    data : pd.DataFrame
        Analysis-ready dataframe with canopy_layer column
    figsize : tuple
        Figure size
    
    Returns
    -------
    matplotlib.figure.Figure
        The created figure
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    colors = {'Lower': 'forestgreen', 'Middle': 'orange', 'Upper': 'crimson'}
    
    for i, layer in enumerate(['Lower', 'Middle', 'Upper']):
        subset = data[data['canopy_layer'] == layer]
        
        axes[i].hexbin(subset['humid_temp'], subset['humidity'], 
                       gridsize=30, cmap='YlOrRd', mincnt=1, alpha=0.8)
        axes[i].set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
        axes[i].set_ylabel('Relative Humidity (%)', fontsize=12, fontweight='bold')
        axes[i].set_title(f'{layer} Canopy', fontsize=14, fontweight='bold', 
                         color=colors[layer])
        axes[i].grid(True, alpha=0.3)
        
        # Add sample size
        n = len(subset)
        axes[i].text(0.05, 0.95, f'n = {n:,}', transform=axes[i].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    return fig


def plot_data_cleaning_summary(original_data, cleaned_data, figsize=(14, 5)):
    """
    Visualize the impact of data cleaning
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Fix voltage in original data for plotting (same logic as cleaning)
    original_voltage_fixed = original_data['voltage'].apply(
        lambda v: v / 1023 * 3.3 if v > 10 else v
    )
    
    # Voltage distribution
    axes[0].hist(original_voltage_fixed.dropna(), bins=50, alpha=0.6, 
                label='Original', color='red', edgecolor='darkred')
    axes[0].hist(cleaned_data['voltage'].dropna(), bins=50, alpha=0.6, 
                label='Cleaned', color='green', edgecolor='darkgreen')
    axes[0].axvline(2.4, color='black', linestyle='--', linewidth=2, label='Min threshold (2.4V)')
    axes[0].axvline(3.0, color='black', linestyle='--', linewidth=2, label='Max threshold (3.0V)')
    axes[0].set_xlabel('Voltage (V)', fontsize=12, fontweight='bold')
    axes[0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[0].set_title('Voltage Distribution', fontsize=14, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Temperature distribution
    axes[1].hist(original_data['humid_temp'].dropna(), bins=50, alpha=0.6, 
                label='Original', color='red', edgecolor='darkred')
    axes[1].hist(cleaned_data['humid_temp'].dropna(), bins=50, alpha=0.6, 
                label='Cleaned', color='green', edgecolor='darkgreen')
    axes[1].set_xlabel('Temperature (°C)', fontsize=12, fontweight='bold')
    axes[1].set_ylabel('Frequency', fontsize=12, fontweight='bold')
    axes[1].set_title('Temperature Distribution', fontsize=14, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig