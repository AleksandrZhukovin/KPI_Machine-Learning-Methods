import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
import matplotlib


matplotlib.use('TkAgg')

# Dataset: https://www.kaggle.com/datasets/doaaalsenani/usa-cers-dataset
df = pd.read_csv('data.csv')[['year', 'mileage']]


def data_prep_and_check():
    global df
    z_scores = np.abs(stats.zscore(df))
    anomalies = (z_scores > 7).any(axis=1)

    _, axes = plt.subplots(1, 3, figsize=(12, 6))

    axes[0].scatter(df['year'], df['mileage'])
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Mileage')
    axes[0].set_title('Original')
    axes[0].grid(True)

    axes[1].scatter(df[~anomalies]['year'], df[~anomalies]['mileage'], label='Normal')
    axes[1].scatter(df[anomalies]['year'], df[anomalies]['mileage'], color='red', label='Anomalies')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Mileage')
    axes[1].set_title('Anomalies')
    axes[1].legend()
    axes[1].grid(True)

    df = df[~anomalies]
    correlation = df.corr()

    sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                square=True, cbar_kws={'shrink': 0.8}, ax=axes[2])
    axes[2].set_title('Correlation')

    plt.show()


def scikit_model():
    global df
    X = df[['year']]
    y = df['mileage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    _, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    axes[0].scatter(X_train, y_train)
    X_range = np.linspace(X['year'].min(), X['year'].max(), 100).reshape(-1, 1)
    y_range = model.predict(X_range)
    axes[0].plot(X_range, y_range, color='red', linewidth=2)
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Mileage')
    axes[0].set_title('Scikit Regression')
    axes[0].grid(True)
    
    axes[1].scatter(X_test, y_test)
    axes[1].scatter(X_test, y_pred, color='red')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Mileage')
    axes[1].set_title('Scikit Prediction')
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.show()


def xg_boost():
    global df
    X = df[['year']]
    y = df['mileage']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    model = XGBRegressor(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        random_state=42
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(X_train, y_train)
    X_range = np.linspace(X['year'].min(), X['year'].max(), 200).reshape(-1, 1)
    y_range = model.predict(X_range)
    axes[0].plot(X_range, y_range, color='red', linewidth=2)
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Mileage')
    axes[0].set_title('XGBoost Regression')
    axes[0].grid(True)

    axes[1].scatter(X_test, y_test, label='True')
    axes[1].scatter(X_test, y_pred, color='red', label='Predicted')
    axes[1].set_xlabel('Year')
    axes[1].set_ylabel('Mileage')
    axes[1].set_title('XGBoost Prediction')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def vanilla_regression():
    global df
    X = df[['year']].values
    y = df['mileage'].values
    X_norm = (X - X.mean()) / X.std()
    X_b = np.c_[np.ones((X_norm.shape[0], 1)), X_norm]
    theta = np.random.randn(2, 1)
    lr = 0.01
    n_iter = 1000
    m = len(X_b)

    for _ in range(n_iter):
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y.reshape(-1, 1))
        theta -= lr * gradients

    y_pred = X_b.dot(theta)

    _, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].scatter(X, y, label="Data")
    X_range = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)
    X_range_norm = (X_range - X.mean()) / X.std()
    X_range_b = np.c_[np.ones((X_range_norm.shape[0], 1)), X_range_norm]
    y_range = X_range_b.dot(theta)
    axes[0].plot(X_range, y_range, color="red", linewidth=2, label="Custom Regression")
    axes[0].set_xlabel("Year")
    axes[0].set_ylabel("Mileage")
    axes[0].set_title("Vanilla Regression")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].scatter(X, y, label="True")
    axes[1].scatter(X, y_pred, color="red", label="Predicted")
    axes[1].set_xlabel("Year")
    axes[1].set_ylabel("Mileage")
    axes[1].set_title("Vanilla Prediction")
    axes[1].legend()
    axes[1].grid(True)
    plt.tight_layout()
    plt.show()


data_prep_and_check()
scikit_model()
xg_boost()
vanilla_regression()
