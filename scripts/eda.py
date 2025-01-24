import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def overview_data(data):
    print(data.info())
    print(data.describe())
    return overview_data

def plot_numerical_distribution(data):
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    for feature in numerical_features:
        sns.histplot(data[feature], kde=True)
        plt.title(f'Distribution of {feature}')
        plt.show()

def plot_categorical_distribution(data):
    categorical_features = data.select_dtypes(include=['object']).columns
    for feature in categorical_features:
        sns.countplot(x=data[feature])
        plt.title(f'Distribution of {feature}')
        plt.show()

def correlation_analysis(data):
    corr = data.corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title('Correlation Matrix')
    plt.show()

def identify_missing_values(data):
    missing_values = data.isnull().sum()
    print(missing_values[missing_values > 0])

def detect_outliers(data):
    numerical_features = data.select_dtypes(include=['int64', 'float64']).columns
    for feature in numerical_features:
        sns.boxplot(x=data[feature])
        plt.title(f'Boxplot of {feature}')
        plt.show()