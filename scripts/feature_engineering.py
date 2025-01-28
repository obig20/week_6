import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

# Function to create aggregate features based on CustomerId
def create_aggregate_features(data):
    if 'CustomerId' in data.columns and 'Amount' in data.columns:
        data['Total_Transaction_Amount'] = data.groupby('CustomerId')['Amount'].transform('sum')
        data['Average_Transaction_Amount'] = data.groupby('CustomerId')['Amount'].transform('mean')
        data['Transaction_Count'] = data.groupby('CustomerId')['Amount'].transform('count')
        data['Std_Transaction_Amount'] = data.groupby('CustomerId')['Amount'].transform('std')
    else:
        raise KeyError("Required columns 'CustomerId' or 'Amount' are missing from the DataFrame")
    return data

# Function to extract date-time features from TransactionStartTime
def extract_features(data):
    data['Transaction_Hour'] = pd.to_datetime(data['TransactionStartTime']).dt.hour
    data['Transaction_Day'] = pd.to_datetime(data['TransactionStartTime']).dt.day
    data['Transaction_Month'] = pd.to_datetime(data['TransactionStartTime']).dt.month
    data['Transaction_Year'] = pd.to_datetime(data['TransactionStartTime']).dt.year
    return data

# Function to encode categorical variables using OneHotEncoder
from sklearn.preprocessing import OneHotEncoder ,LabelEncoder
import pandas as pd
from scipy.sparse import csr_matrix

def one_hot_encode(df, column):
    encoder = OneHotEncoder(sparse=False)
    encoded = encoder.fit_transform(df[[column]])
    encoded_df = pd.DataFrame(encoded, columns=encoder.get_feature_names_out([column]))
    return pd.concat([df, encoded_df], axis=1).drop(column, axis=1)

def label_encode(df, column):
    encoder = LabelEncoder()
    df[column] = encoder.fit_transform(df[column])
    return df


# Function to handle missing values using SimpleImputer

import pandas as pd
from sklearn.impute import SimpleImputer

import pandas as pd
from sklearn.impute import SimpleImputer

import pandas as pd
from sklearn.impute import SimpleImputer

def handle_missing_values(data, strategy='mean'):
    # Separate numeric and non-numeric columns
    numeric_cols = data.select_dtypes(include=['number']).columns
    non_numeric_cols = data.select_dtypes(exclude=['number']).columns
    
    # Handle missing values for numeric columns
    if strategy in ['mean', 'median', 'most_frequent']:
        imputer = SimpleImputer(strategy=strategy)
        data[numeric_cols] = imputer.fit_transform(data[numeric_cols])
    else:
        raise ValueError("Invalid strategy. Use 'mean', 'median', or 'most_frequent'.")
    
    # Handle missing values for non-numeric columns
    imputer = SimpleImputer(strategy='most_frequent')
    data[non_numeric_cols] = imputer.fit_transform(data[non_numeric_cols])
    
    return data
# Function to normalize and standardize numerical features using StandardScaler
def normalize_standardize_features(data):
    scaler = StandardScaler()
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data

