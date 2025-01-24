import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer

def create_aggregate_features(data):
    data['Total_Transaction_Amount'] = data.groupby('customer_id')['transaction_amount'].transform('sum')
    data['Average_Transaction_Amount'] = data.groupby('customer_id')['transaction_amount'].transform('mean')
    data['Transaction_Count'] = data.groupby('customer_id')['transaction_amount'].transform('count')
    data['Std_Transaction_Amount'] = data.groupby('customer_id')['transaction_amount'].transform('std')
    return data

def extract_features(data):
    data['Transaction_Hour'] = pd.to_datetime(data['transaction_time']).dt.hour
    data['Transaction_Day'] = pd.to_datetime(data['transaction_time']).dt.day
    data['Transaction_Month'] = pd.to_datetime(data['transaction_time']).dt.month
    data['Transaction_Year'] = pd.to_datetime(data['transaction_time']).dt.year
    return data

def encode_categorical_variables(data):
    encoder = OneHotEncoder(sparse=False)
    categorical_features = data.select_dtypes(include=['object']).columns
    encoded_data = pd.DataFrame(encoder.fit_transform(data[categorical_features]))
    encoded_data.columns = encoder.get_feature_names_out(categorical_features)
    data = data.drop(categorical_features, axis=1)
    data = data.join(encoded_data)
    return data

def handle_missing_values(data):
    imputer = SimpleImputer(strategy='mean')
    data = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)
    return data

def normalize_standardize_features(data):
    scaler = StandardScaler()
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    data[numerical_features] = scaler.fit_transform(data[numerical_features])
    return data