import pytest
import numpy as np
import pandas as pd
from src.model import sigmoid, cost_function, log_fit, predict

def test_sigmoid():
    """Verify the sigmoid function maps values correctly to (0, 1)."""
    assert sigmoid(0) == 0.5
    assert sigmoid(10) > 0.99
    assert sigmoid(-10) < 0.01
    # Check that it handles numpy arrays
    arr = np.array([0, 2])
    assert np.allclose(sigmoid(arr), [0.5, 1 / (1 + np.exp(-2))])

def test_cost_function():
    """Ensure cost function returns a float and handles perfect vs bad predictions."""
    X = np.array([[1, 2], [3, 4]])
    y = np.array([1, 0])
    # weights for 2 features + 1 bias = 3 weights
    weights = np.zeros(3)
    
    cost = cost_function(X, y, weights)
    assert isinstance(cost, float)
    assert cost > 0  # Initial cost with zero weights should be positive

def test_predict_shapes():
    """Ensure the predict function returns the correct dimensions."""
    X = np.random.rand(10, 3)  # 10 rows, 3 features
    weights = np.zeros(4)      # 3 features + 1 bias
    
    predictions = predict(X, weights)
    assert predictions.shape == (10,)
    assert set(np.unique(predictions)).issubset({0, 1})

def test_model_convergence():
    """A functional test to ensure weights update and cost decreases."""
    # Create a simple linearly separable dataset
    X = np.array([[1], [5]])
    y = np.array([0, 1])
    
    # Run a very short training session
    initial_weights, initial_cost_hist = log_fit(X, y, learning_rate=0.1, iterations=1)
    updated_weights, final_cost_hist = log_fit(X, y, learning_rate=0.1, iterations=200)
    
    # The cost after 200 iterations should be lower than the first iteration
    assert final_cost_hist[-1] < initial_cost_hist[0]

def test_data_separation_logic():
    """Verify that 'forbidden' leakage columns are actually dropped."""
    from src.model import seperate_features_and_target
    
    data = {
        'median_age': [30, 40],
        'facility_count': [1, 5],
        'facility_density': [0.1, 0.5],
        'zip_code': ['77001', '77002'],
        'is_desert': [1, 0]
    }
    df = pd.DataFrame(data)
    X, y = seperate_features_and_target(df, 'is_desert')
    
    # X should only have 1 column (median_age) because the others are forbidden
    assert X.shape[1] == 1
    assert y.shape[0] == 2