import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Simulate a Dataset
def simulate_data():
    np.random.seed(42)  # Seed for reproducibility
    X = 2 * np.random.rand(100, 1)
    y = 4 + 3 * X + np.random.randn(100, 1)
    return X, y

# Linear Regression and Plotting
def plot_regression(X, y):
    reg = LinearRegression().fit(X, y)
    line = reg.coef_ * X + reg.intercept_

    plt.scatter(X, y)
    plt.plot(X, line, 'r')
    plt.title("Linear Regression")
    plt.xlabel("X")
    plt.ylabel("y")
    st.pyplot()
    
    return reg  # Return the regression object

st.title("Interactive Regression on Simulated Data")

# Simulate the data
X, y = simulate_data()

if st.button('Run Regression'):
    reg = plot_regression(X, y)  # Capture the returned regression object
    st.write(f"Equation: y = {reg.coef_[0][0]:.2f}X + {reg.intercept_[0]:.2f}")
