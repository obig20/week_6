from scripts.eda import overview_data, plot_numerical_distribution, plot_categorical_distribution, correlation_analysis, identify_missing_values, detect_outliers
from scripts.feature_engineering import create_aggregate_features, extract_features, encode_categorical_variables, handle_missing_values, normalize_standardize_features
from scripts.modeling import split_data, train_model, evaluate_model
from scripts.utils import load_data

file_path = r'C:\Users\h\Desktop\week 6\data\credit_data.csv'

# Load data
data = load_data(file_path)

# EDA
overview_data(data)
plot_numerical_distribution(data)
plot_categorical_distribution(data)
correlation_analysis(data)
identify_missing_values(data)
detect_outliers(data)

# Feature Engineering
data = create_aggregate_features(data)
data = extract_features(data)
data = encode_categorical_variables(data)
data = handle_missing_values(data)
data = normalize_standardize_features(data)

# Prepare data for modeling
X = data.drop('target', axis=1)
y = data['target']
X_train, X_test, y_train, y_test = split_data(X, y)

# Train and evaluate models
model = train_model(X_train, y_train, model_type='logistic')
evaluate_model(model, X_test, y_test)

# Save the model
import joblib
joblib.dump(model, 'model.pkl')