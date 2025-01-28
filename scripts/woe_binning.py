import pandas as pd
import numpy as np

def woe_binning(data, target, feature, n_bins=10):
    """Perform WoE binning on a feature."""
    data['bins'] = pd.qcut(data[feature], q=n_bins, duplicates='drop')
    woe_df = data.groupby('bins')[target].agg(['count', 'sum'])
    woe_df['non_events'] = woe_df['count'] - woe_df['sum']
    woe_df['event_rate'] = woe_df['sum'] / woe_df['sum'].sum()
    woe_df['non_event_rate'] = woe_df['non_events'] / woe_df['non_events'].sum()
    woe_df['woe'] = np.log(woe_df['event_rate'] / woe_df['non_event_rate'])
    woe_df['iv'] = (woe_df['event_rate'] - woe_df['non_event_rate']) * woe_df['woe']
    
    return woe_df

def calculate_iv(data, target, features, n_bins=10):
    """Calculate Information Value (IV) for multiple features."""
    iv_dict = {}
    for feature in features:
        woe_df = woe_binning(data, target, feature, n_bins)
        iv_dict[feature] = woe_df['iv'].sum()
    
    return iv_dict
import seaborn as sns
import matplotlib.pyplot as plt

def plot_woe(woe_df, feature):
    """Plot the WoE values for a feature."""
    plt.figure(figsize=(10, 6))
    sns.barplot(x=woe_df.index.astype(str), y=woe_df['woe'])
    plt.title(f'WoE for {feature}')
    plt.xticks(rotation=45)
    plt.xlabel('Bins')
    plt.ylabel('WoE')
    plt.show()
