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
def encode_categorical_variables(data):
    encoder = OneHotEncoder(sparse=False)
    categorical_features = data.select_dtypes(include=['object']).columns
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_features]))
    encoded_data.columns = encoder.get_feature_names_out(categorical_features)
    data = data.drop(categorical_features, axis=1)
    data = data.join(encoded_data)
    return data

# Function to handle missing values using SimpleImputer
def handle_missing_values(data):
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data

# Function to normalize and standardize numerical features using StandardScaler
def normalize_standardize_features(data):
    scaler = StandardScaler()
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data