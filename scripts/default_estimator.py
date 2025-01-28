import numpy as np
import pandas as pd 
from sklearn.preprocessing import StandardScaler

def calculate_rfms_scores(data):
    """Calculate RFMS scores based on the transaction data."""
    # Example derived features
    data['Recency'] = (pd.to_datetime('now') - pd.to_datetime(data['TransactionStartTime'])).dt.days
    data['Frequency'] = data.groupby('CustomerId')['TransactionId'].transform('count')
    data['Monetary'] = data.groupby('CustomerId')['Amount'].transform('sum')
    data['Stability'] = data.groupby('CustomerId')['Amount'].transform('std').fillna(0)
    
    # Normalize the scores
    scaler = StandardScaler()
    rfms_scores = scaler.fit_transform(data[['Recency', 'Frequency', 'Monetary', 'Stability']])
    
    return rfms_scores, data
from sklearn.cluster import KMeans

def classify_users(rfms_scores):
    """Classify users into high and low RFMS scores using KMeans clustering."""
    kmeans = KMeans(n_clusters=2, random_state=42)
    labels = kmeans.fit_predict(rfms_scores)
    return labels
