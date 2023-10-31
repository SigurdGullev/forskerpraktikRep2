import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics.regressionplots as sm

st.title("Directed Acyclical Graphs (DAGs)")

# Create four buttons in a row
buttons = st.columns(4)

button_labels = ['Generate Collider DAG', 'Generate Mediator DAG', 'Generate RCT DAG', 'Generate Confounding DAG']
button_functions = [None, None, None, None]

for i, button_label in enumerate(button_labels):
    with buttons[i]:
        button_functions[i] = st.button(button_label)

# Function to plot with a calculated regression line
def plot_with_regression_line(df, x_col, y_col, title):
    x = df[x_col]
    y = df[y_col]
    coefficients = np.polyfit(x, y, 1)  # Fit a linear regression model
    regression_line = np.polyval(coefficients, x)  # Calculate the regression line
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(df[x_col], df[y_col], alpha=0.5)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    
    ax.plot(x, regression_line, color='black', linewidth=1, label='Regression Line')
    ax.legend()
    st.pyplot(fig)

# DAG simulation and plot functions
def simulate_data(size, x_coeff, z_coeff, y_coeff):
    X = np.random.normal(size=size)
    Z = x_coeff * X + np.random.normal(size=size)
    e = np.random.normal(size=size)
    Y = y_coeff * Z + e
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df

def plot_dag(df, title, x_col, y_col):
    plot_with_regression_line(df, x_col, y_col, title)

    # Partial regression with Z as a control variable
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog=y_col, exog_i=x_col, exog_others=['Z'], data=df, ax=ax, obs_labels=False)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    st.pyplot(fig)

# Button actions
if button_functions[0]:
    df = simulate_data(1000, 2, 1, 1)
    plot_dag(df, "Collider DAG", "X", "Y")

if button_functions[1]:
    df = simulate_data(1000, 1.5, 1, 2)
    plot_dag(df, "Mediator DAG", "X", "Y")

if button_functions[2]:
    df = simulate_data(1000, 1.5, 2, 1)
    plot_dag(df, "RCT DAG", "X", "Y")

if button_functions[3]:
    df = simulate_data(1000, 1.5, 2, 1.3)
    plot_dag(df, "Confounding DAG", "X", "Y")
