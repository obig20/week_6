import numpy as np
import pandas as pd
from scripts.default_estimator import calculate_rfms_scores
from scripts.default_estimator import classify_users
from scripts.woe_binning import calculate_iv, woe_binning
from scripts.woe_binning import plot_woe

def main():
    # Load and preprocess data
    file_path = '../data/data.csv'
    data = pd.read_csv(file_path)
    
    # Calculate RFMS scores and classify users
    rfms_scores, data = calculate_rfms_scores(data)
    labels = classify_users(rfms_scores)
    data['RFMS_Label'] = labels
    
    # Assign good and bad labels based on RFMS scores
    data['Default_Proxy'] = np.where(data['RFMS_Label'] == 1, 'Good', 'Bad')
    
    # Perform WoE binning
    target = 'FraudResult'  # Assuming FraudResult is the target variable
    features = ['Recency', 'Frequency', 'Monetary', 'Stability']
    iv_dict = calculate_iv(data, target, features)
    
    # Print IV values
    for feature, iv in iv_dict.items():
        print(f"IV for {feature}: {iv}")
    
    # Plot WoE for a feature
    woe_df = woe_binning(data, target, features[0])
    plot_woe(woe_df, features[0])

if __name__ == "__main__":
    main()
