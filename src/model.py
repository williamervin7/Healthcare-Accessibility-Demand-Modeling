import sys
import os


sys.path.append(os.path.abspath("."))
from src.preprocessing import filter_by_city
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

def sigmoid(x):
    """
    Compute the sigmoid of z.
    
    Args:
        z (ndarray): A scalar or numpy array of any size.
        
    Returns:
        ndarray: sigmoid(z), values between 0 and 1.
    """
    return 1 / (1 + np.exp(-x))

def cost_function(X, y, weights):
    """
    Compute the Binary Cross-Entropy loss for Logistic Regression.
    
    Args:
        X (ndarray): Feature matrix (m x n).
        y (ndarray): Target labels (m x 1).
        weights (ndarray): Model parameters (n+1 x 1).
        
    Returns:
        float: The average log-loss for the current parameters.
    """
    m = len(y)
    # Add a column of ones to X to account for the bias/intercept
    X_bias = np.c_[np.ones((m, 1)), X]
    h = sigmoid(np.dot(X_bias, weights))

    epsilon = 1e-5  # To prevent log(0)
    cost = (1/m) * np.sum(-y * np.log(h + epsilon) - (1 - y) * np.log(1 - h + epsilon))
    return cost

def log_fit(X, y, learning_rate=0.01, iterations=1000):
    """
    Train a Logistic Regression model using Gradient Descent.
    
    Args:
        X (ndarray): Scaled feature matrix.
        y (ndarray): Binary target labels.
        learning_rate (float): Step size for weight updates.
        iterations (int): Number of times to loop through the data.
        
    Returns:
        tuple: (final_weights, cost_history)
    """
    m = len(y)
    # 1. Add bias column to X (The "Matrix Trick")
    X_bias = np.c_[np.ones((m, 1)), X]
    
    # 2. Initialize weights (n_features + 1 for the bias)
    weights = np.zeros(X_bias.shape[1])
    cost_history = []

    # 3. Training Loop
    for i in range(iterations):
        # Calculate predictions
        h = sigmoid(np.dot(X_bias, weights))
        
        # Calculate gradient
        gradient = np.dot(X_bias.T, (h - y)) / m
        
        # Update weights (The core Gradient Descent step)
        weights -= learning_rate * gradient
        
        # Track cost
        if i % 100 == 0:
            cost = cost_function(X, y, weights)
            cost_history.append(cost)
            print(f"Iteration {i}: Cost {cost}")

    return weights, cost_history

def predict(X, weights):
    """
    Classify input data based on trained weights.
    
    Args:
        X (ndarray): Feature matrix for prediction.
        weights (ndarray): Trained model weights including bias.
        
    Returns:
        ndarray: Binary predictions (0 or 1).
    """
    X_bias = np.c_[np.ones((X.shape[0], 1)), X]
    probabilities = sigmoid(np.dot(X_bias, weights))
    return (probabilities >= 0.5).astype(int)

def seperate_features_and_target(df: pd.DataFrame, target_col: str):
    """
    Isolate features from the target and remove columns that cause data leakage.
    
    Args:
        df (pd.DataFrame): The full processed dataset.
        target_col (str): The name of the desert classification column.
        
    Returns:
        tuple: (X, y) as numpy arrays.
    """
    # Columns that directly measure supply must be dropped to prevent leakage
    forbidden_columns = [
        target_col, 
        'facility_count', 
        'facility_density', 
        'zip_code'
    ]
    
    X = df.drop(columns=forbidden_columns).values
    y = df[target_col].values
    return X, y


def scale_features(X_train, X_test):
    """
    Standardize features by removing the mean and scaling to unit variance.
    
    Note: The scaler is fit only on the training set to prevent data leakage.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    return X_train_scaled, X_test_scaled

def run_model(df: pd.DataFrame, target_col: str):
    """
    Orchestrate the full training and evaluation pipeline for the custom model.
    """
    X, y = seperate_features_and_target(df, target_col)

    # Split: 80% Training, 20% Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    print("--- Training Custom Model ---")
    weights, cost_history = log_fit(X_train_scaled, y_train)
    
    y_pred = predict(X_test_scaled, weights)
    
    print("Classification Report Custom Model:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return weights, cost_history, y_test, y_pred

def run_sklearn_model(df: pd.DataFrame, target_col: str):
    """
    Run Scikit-learn's Logistic Regression for benchmarking results.
    """
    X, y = seperate_features_and_target(df, target_col)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train_scaled, X_test_scaled = scale_features(X_train, X_test)
    
    model = LogisticRegression()
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    
    print("Classification Report Scikit-learn Model:")
    print(classification_report(y_test, y_pred))
    
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    return y_test, y_pred


if __name__ == "__main__":
    houston_df = pd.read_csv('data/processed_data.csv', dtype={'zip_code': str})

    houston_df = filter_by_city(houston_df, 'houston')
    run_model(houston_df, target_col='is_desert')
    run_sklearn_model(houston_df, target_col='is_desert')
