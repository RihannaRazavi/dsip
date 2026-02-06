import pandas as pd
import numpy as np

def find_empirical_voltage_conversion(net_data, log_data):
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\n=== EMPIRICAL VOLTAGE ALIGNMENT ===")
    net_voltages = net_data['voltage'].dropna().values
    log_voltages = log_data['voltage'].dropna().values
    
    print(f"Net: n={len(net_voltages):,}, mean={net_voltages.mean():.2f}, range=[{net_voltages.min():.2f}, {net_voltages.max():.2f}]")
    print(f"Log: n={len(log_voltages):,}, mean={log_voltages.mean():.2f}, range=[{log_voltages.min():.2f}, {log_voltages.max():.2f}]")

    print("\n=== METHOD 1: Percentile Matching ===")
    
    net_p25, net_p50, net_p75 = np.percentile(net_voltages, [25, 50, 75])
    log_p25, log_p50, log_p75 = np.percentile(log_voltages, [25, 50, 75])
    
    print(f"Net percentiles: 25%={net_p25:.1f}, 50%={net_p50:.1f}, 75%={net_p75:.1f}")
    print(f"Log percentiles: 25%={log_p25:.2f}, 50%={log_p50:.2f}, 75%={log_p75:.2f}")
    net_percentiles = np.array([net_p25, net_p50, net_p75])
    log_percentiles = np.array([log_p25, log_p50, log_p75])
    
    coeffs = np.polyfit(net_percentiles, log_percentiles, 1)
    scale = coeffs[0]
    offset = coeffs[1]
    
    print(f"\nLinear fit: log_V = {scale:.6f} * net_V + {offset:.6f}")
    print("\n=== METHOD 2: Range Matching ===")
    net_range = net_voltages.max() - net_voltages.min()
    log_range = log_voltages.max() - log_voltages.min()
    scale_range = log_range / net_range
    offset_range = log_voltages.min() - (net_voltages.min() * scale_range)
    
    print(f"Range matching: log_V = {scale_range:.6f} * net_V + {offset_range:.6f}")
    
    conversion_factor = scale
    conversion_offset = offset
    
    # Apply conversion
    net_converted = net_voltages * conversion_factor + conversion_offset
    
    print(f"\n=== AFTER CONVERSION ===")
    print(f"Net converted: mean={net_converted.mean():.2f}, range=[{net_converted.min():.2f}, {net_converted.max():.2f}]")
    print(f"Log original:  mean={log_voltages.mean():.2f}, range=[{log_voltages.min():.2f}, {log_voltages.max():.2f}]")
    print(f"Mean difference: {abs(net_converted.mean() - log_voltages.mean()):.3f}V")
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    axes[0, 0].hist(net_voltages, bins=50, alpha=0.7, color='blue', edgecolor='darkblue')
    axes[0, 0].set_xlabel('Net Voltage (ADC)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title(f'Net Original\nmean={net_voltages.mean():.1f}', fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(log_voltages, bins=50, alpha=0.7, color='green', edgecolor='darkgreen')
    axes[0, 1].set_xlabel('Log Voltage (V)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title(f'Log Original\nmean={log_voltages.mean():.2f}V', fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 2].scatter(net_percentiles, log_percentiles, s=100, color='red', zorder=3, label='Percentiles (25, 50, 75)')
    axes[0, 2].plot(net_percentiles, scale * net_percentiles + offset, 'r--', linewidth=2, label=f'Fit: y={scale:.4f}x+{offset:.4f}')
    axes[0, 2].set_xlabel('Net Voltage (ADC)', fontsize=11, fontweight='bold')
    axes[0, 2].set_ylabel('Log Voltage (V)', fontsize=11, fontweight='bold')
    axes[0, 2].set_title('Percentile Matching', fontsize=12, fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    axes[1, 0].hist(net_converted, bins=50, alpha=0.6, color='blue', edgecolor='darkblue', label='Net (converted)')
    axes[1, 0].hist(log_voltages, bins=50, alpha=0.6, color='green', edgecolor='darkgreen', label='Log (original)')
    axes[1, 0].set_xlabel('Voltage (V)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('Full Range Overlay', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)


    net_valid = net_converted[(net_converted >= 0) & (net_converted <= 4)]
    log_valid = log_voltages[(log_voltages >= 0) & (log_voltages <= 4)]
    
    axes[1, 1].hist(net_valid, bins=40, alpha=0.6, color='blue', edgecolor='darkblue', label=f'Net (n={len(net_valid):,})')
    axes[1, 1].hist(log_valid, bins=40, alpha=0.6, color='green', edgecolor='darkgreen', label=f'Log (n={len(log_valid):,})')
    axes[1, 1].set_xlabel('Voltage (V)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 1].set_xlim(0, 4)
    axes[1, 1].set_title('Valid Range (0-4V)', fontsize=12, fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    # Q-Q plot to check alignment
    net_sorted = np.sort(net_converted)
    log_sorted = np.sort(log_voltages)
    
    n_points = min(len(net_sorted), len(log_sorted))
    net_qq = np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(net_sorted)), net_sorted)
    log_qq = np.interp(np.linspace(0, 1, n_points), np.linspace(0, 1, len(log_sorted)), log_sorted)
    
    axes[1, 2].scatter(log_qq, net_qq, alpha=0.3, s=5, color='purple')
    axes[1, 2].plot([log_qq.min(), log_qq.max()], [log_qq.min(), log_qq.max()], 'r--', linewidth=2, label='Perfect match')
    axes[1, 2].set_xlabel('Log Voltage Quantiles (V)', fontsize=11, fontweight='bold')
    axes[1, 2].set_ylabel('Net Converted Quantiles (V)', fontsize=11, fontweight='bold')
    axes[1, 2].set_title('Q-Q Plot (Distribution Alignment)', fontsize=12, fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return conversion_factor, conversion_offset, fig




def find_voltage_conversion_factor(net_data, log_data):
    
    import matplotlib.pyplot as plt
    import numpy as np
    
    print("\n=== VOLTAGE DISTRIBUTION ANALYSIS ===")
    net_voltages = net_data['voltage'].dropna()
    log_voltages = log_data['voltage'].dropna()
    
    net_mean = net_voltages.mean()
    net_max = net_voltages.max()
    net_min = net_voltages.min()
    
    log_mean = log_voltages.mean()
    log_max = log_voltages.max()
    log_min = log_voltages.min()
    
    print(f"Net voltage: mean={net_mean:.2f}, min={net_min:.2f}, max={net_max:.2f}")
    print(f"Log voltage: mean={log_mean:.2f}, min={log_min:.2f}, max={log_max:.2f}")
    if net_max > 100 and log_max < 10:
        print("\nNet dataset: Raw ADC values (0-1023)")
        print("Log dataset: Actual volts (0-3.3V)")
        conversion_factor = 3.3 / 1023.0
        net_converted = net_voltages * conversion_factor
        
        which_to_convert = 'net'
        
    elif log_max > 100 and net_max < 10:
        conversion_factor = 3.3 / 1023.0
        log_converted = log_voltages * conversion_factor
        
        
        which_to_convert = 'log'
        
    else:
        which_to_convert = 'none'
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes[0, 0].hist(net_voltages, bins=50, alpha=0.7, color='blue', edgecolor='darkblue')
    axes[0, 0].set_xlabel('Net Voltage (original scale)', fontsize=11, fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 0].set_title(f'Net Dataset Original\nmean={net_mean:.1f}, max={net_max:.1f}', 
                        fontsize=12, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 1].hist(log_voltages, bins=50, alpha=0.7, color='green', edgecolor='darkgreen')
    axes[0, 1].set_xlabel('Log Voltage (original scale)', fontsize=11, fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[0, 1].set_title(f'Log Dataset Original\nmean={log_mean:.1f}, max={log_max:.1f}', 
                        fontsize=12, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    if which_to_convert == 'net':
        net_converted = net_voltages * conversion_factor
        axes[1, 0].hist(net_converted, bins=50, alpha=0.6, color='blue', 
                       edgecolor='darkblue', label='Net (converted)')
        axes[1, 0].hist(log_voltages, bins=50, alpha=0.6, color='green', 
                       edgecolor='darkgreen', label='Log (original)')
    elif which_to_convert == 'log':
        log_converted = log_voltages * conversion_factor
        axes[1, 0].hist(net_voltages, bins=50, alpha=0.6, color='blue', 
                       edgecolor='darkblue', label='Net (original)')
        axes[1, 0].hist(log_converted, bins=50, alpha=0.6, color='green', 
                       edgecolor='darkgreen', label='Log (converted)')
    else:
        axes[1, 0].hist(net_voltages, bins=50, alpha=0.6, color='blue', 
                       edgecolor='darkblue', label='Net')
        axes[1, 0].hist(log_voltages, bins=50, alpha=0.6, color='green', 
                       edgecolor='darkgreen', label='Log')
    
    axes[1, 0].set_xlabel('Voltage (V)', fontsize=11, fontweight='bold')
    axes[1, 0].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 0].set_title('After Conversion - Full Range', fontsize=12, fontweight='bold')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    if which_to_convert == 'net':
        net_valid = net_converted[(net_converted >= 0) & (net_converted <= 4)]
        log_valid = log_voltages[(log_voltages >= 0) & (log_voltages <= 4)]
        axes[1, 1].hist(net_valid, bins=40, alpha=0.6, color='blue', 
                       edgecolor='darkblue', label=f'Net (n={len(net_valid):,})')
        axes[1, 1].hist(log_valid, bins=40, alpha=0.6, color='green', 
                       edgecolor='darkgreen', label=f'Log (n={len(log_valid):,})')
    elif which_to_convert == 'log':
        net_valid = net_voltages[(net_voltages >= 0) & (net_voltages <= 4)]
        log_valid = log_converted[(log_converted >= 0) & (log_converted <= 4)]
        axes[1, 1].hist(net_valid, bins=40, alpha=0.6, color='blue', 
                       edgecolor='darkblue', label=f'Net (n={len(net_valid):,})')
        axes[1, 1].hist(log_valid, bins=40, alpha=0.6, color='green', 
                       edgecolor='darkgreen', label=f'Log (n={len(log_valid):,})')
    else:
        net_valid = net_voltages[(net_voltages >= 0) & (net_voltages <= 4)]
        log_valid = log_voltages[(log_voltages >= 0) & (log_voltages <= 4)]
        axes[1, 1].hist(net_valid, bins=40, alpha=0.6, color='blue', 
                       edgecolor='darkblue', label=f'Net (n={len(net_valid):,})')
        axes[1, 1].hist(log_valid, bins=40, alpha=0.6, color='green', 
                       edgecolor='darkgreen', label=f'Log (n={len(log_valid):,})')
    
    axes[1, 1].axvline(2.4, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Typical min (2.4V)')
    axes[1, 1].axvline(3.0, color='orange', linestyle='--', linewidth=1.5, alpha=0.7, label='Typical max (3.0V)')
    axes[1, 1].set_xlabel('Voltage (V)', fontsize=11, fontweight='bold')
    axes[1, 1].set_ylabel('Frequency', fontsize=11, fontweight='bold')
    axes[1, 1].set_title('Full Voltage Range (0-4V)', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlim(0, 4)
    axes[1, 1].legend(fontsize=9)
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    return conversion_factor, which_to_convert, fig
    
def calibrate_voltage_using_duplicates(net_data, log_data):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    net_copy = net_data.copy()
    log_copy = log_data.copy()
    
    net_copy['result_time'] = pd.to_datetime(net_copy['result_time'], format='mixed', errors='coerce')
    log_copy['result_time'] = pd.to_datetime(log_copy['result_time'], format='mixed', errors='coerce')
    
    
    net_copy['result_time_rounded'] = net_copy['result_time'].dt.round('1min')
    log_copy['result_time_rounded'] = log_copy['result_time'].dt.round('1min')
    net_subset = net_copy[['nodeid', 'result_time_rounded', 'voltage']].copy()
    net_subset = net_subset.dropna(subset=['voltage'])
    net_subset.columns = ['nodeid', 'result_time', 'voltage_net']
    
    log_subset = log_copy[['nodeid', 'result_time_rounded', 'voltage']].copy()
    log_subset = log_subset.dropna(subset=['voltage'])
    log_subset.columns = ['nodeid', 'result_time', 'voltage_log']
    duplicates = net_subset.merge(
        log_subset,
        on=['nodeid', 'result_time'],
        how='inner'
    )
    
    
    if len(duplicates) == 0:
        net_voltages = net_data['voltage'].dropna()
        log_voltages = log_data['voltage'].dropna()
        
        net_mean = net_voltages.mean()
        log_mean = log_voltages.mean()
        net_max = net_voltages.max()
        log_max = log_voltages.max()
        
        print(f"\nNet voltage: mean={net_mean:.2f}, max={net_max:.2f}")
        print(f"Log voltage: mean={log_mean:.2f}, max={log_max:.2f}")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
        
        # Histograms
        ax1.hist(net_voltages[net_voltages < 10], bins=50, alpha=0.6, 
                label=f'Net (mean={net_voltages[net_voltages < 10].mean():.2f}V)', color='blue')
        ax1.hist(log_voltages[log_voltages < 10], bins=50, alpha=0.6,
                label=f'Log (mean={log_voltages[log_voltages < 10].mean():.2f}V)', color='green')
        ax1.set_xlabel('Voltage (V)', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax1.set_title('Voltage Distributions (< 10V)', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.hist(net_voltages[net_voltages >= 10], bins=50, alpha=0.6,
                label=f'Net (mean={net_voltages[net_voltages >= 10].mean():.2f})', color='blue')
        ax2.hist(log_voltages[log_voltages >= 10], bins=50, alpha=0.6,
                label=f'Log (mean={log_voltages[log_voltages >= 10].mean():.2f})', color='green')
        ax2.set_xlabel('Voltage (raw ADC)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax2.set_title('Voltage Distributions (≥ 10)', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Determine conversion factor
        if net_max > 100:
            conversion_factor = 3.3 / 1023.0
        elif log_max > 100:
            conversion_factor = 3.3 / 1023.0
        else:
            conversion_factor = 1.0
        
        return conversion_factor, pd.DataFrame(), fig
    duplicates['voltage_diff'] = abs(duplicates['voltage_net'] - duplicates['voltage_log'])
    different_scales = duplicates[duplicates['voltage_diff'] > 0.5].copy()
    
    if len(different_scales) == 0:
        conversion_factor = 1.0
        
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.text(0.5, 0.5, 'No calibration needed\nVoltages already in same scale', 
               ha='center', va='center', fontsize=16, transform=ax.transAxes)
        ax.axis('off')
        
        return conversion_factor, different_scales, fig
    
    
    net_mean = different_scales['voltage_net'].mean()
    log_mean = different_scales['voltage_log'].mean()
    
    # Determine conversion factor
    if net_mean > log_mean and net_mean > 10:
        conversion_factor = different_scales['voltage_log'].mean() / different_scales['voltage_net'].mean()
        x_label = 'Network Voltage (raw ADC)'
        y_label = 'Log Voltage (volts)'
        x_data = different_scales['voltage_net']
        y_data = different_scales['voltage_log']
    elif log_mean > net_mean and log_mean > 10:
        conversion_factor = different_scales['voltage_net'].mean() / different_scales['voltage_log'].mean()
        x_label = 'Log Voltage (raw ADC)'
        y_label = 'Network Voltage (volts)'
        x_data = different_scales['voltage_log']
        y_data = different_scales['voltage_net']
    else:
        conversion_factor = 1.0
        x_label = 'Network Voltage'
        y_label = 'Log Voltage'
        x_data = different_scales['voltage_net']
        y_data = different_scales['voltage_log']
    
   
    fig, ax = plt.subplots(figsize=(10, 8))
    
    ax.scatter(x_data, y_data, alpha=0.5, s=20, color='steelblue', edgecolors='navy')
    if len(x_data) > 0:
        z = np.polyfit(x_data, y_data, 1)
        p = np.poly1d(z)
        x_line = np.linspace(x_data.min(), x_data.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2, 
               label=f'Linear fit: y = {z[0]:.4f}x + {z[1]:.4f}')
    
    ax.set_xlabel(x_label, fontsize=13, fontweight='bold')
    ax.set_ylabel(y_label, fontsize=13, fontweight='bold')
    ax.set_title('Voltage Calibration Using Duplicate Measurements', 
                fontsize=15, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    textstr = f'n = {len(different_scales):,} duplicate pairs\nConversion factor: {conversion_factor:.6f}'
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
           verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    return conversion_factor, different_scales, fig
def clean_dates_data(dates_data):

    #
    dates_data['day_of_week'] = dates_data['date'].str.extract(r'(Mon|Tue|Wed|Thu|Fri|Sat|Sun)')
    dates_data['time_chr'] = dates_data['date'].str.extract(r'(\d{1,2}:\d{2}:\d{2})')
    dates_data['time'] = pd.to_datetime(dates_data['time_chr'], format='%H:%M:%S')
    dates_data['date_chr'] = dates_data['date'].str.replace(r'\d{1,2}:\d{2}:\d{2} ', '', regex=True)
    dates_data['date'] = pd.to_datetime(dates_data['date_chr'])
    dates_data['datetime'] = pd.to_datetime(dates_data['date_chr'] + ' ' + dates_data['time_chr'])

    return dates_data


def clean_redwood_data(redwood_data):
    redwood_data = redwood_data.drop_duplicates(
        subset=['nodeid', 'result_time', 'epoch'], 
        keep='first'
    ).copy()

    redwood_data['result_time'] = pd.to_datetime(redwood_data['result_time'], format='mixed')
    #def fix_voltage(voltage):
        #if pd.isna(voltage):
            #return voltage
        #if voltage > 10:  
            #return voltage / 1023 * 3.3
        #else: 
            #return voltage
    
    #redwood_data['voltage'] = redwood_data['voltage'].apply(fix_voltage)
    
    # STEP 4: Remove rows with voltage outside acceptable range (2.4 to 3.0 volts)
    redwood_data = redwood_data[
        (redwood_data['voltage'] >= 0) & 
        (redwood_data['voltage'] <= 4)
    ].copy()
    
    # STEP 5: Remove impossible humidity values (should be 0-100%)
    redwood_data = redwood_data[
        (redwood_data['humidity'] >= 0) & 
        (redwood_data['humidity'] <= 100)
    ].copy()
    
    # STEP 6: Remove extreme temperature outliers (reasonable range: -20 to 50°C)
    redwood_data = redwood_data[
        (redwood_data['humid_temp'] >= -20) & 
        (redwood_data['humid_temp'] <= 50)
    ].copy()
    
    # STEP 7: Remove negative PAR values (light can't be negative!)
    if 'hamatop' in redwood_data.columns:
        redwood_data = redwood_data[
            redwood_data['hamatop'] >= 0
        ].copy()
    
    if 'hamabot' in redwood_data.columns:
        redwood_data = redwood_data[
            redwood_data['hamabot'] >= 0
        ].copy()
    
    return redwood_data


def clean_mote_location_data(mote_data):
    # Remove any whitespace from column names
    mote_data.columns = mote_data.columns.str.strip()
    
    # Remove any rows with missing values
    mote_data = mote_data.dropna()
    
    # ID should already be int from loading, but just in case
    if mote_data['ID'].dtype == 'object':
        mote_data['ID'] = mote_data['ID'].astype(int)
    
    return mote_data