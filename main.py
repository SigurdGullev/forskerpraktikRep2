import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics.regressionplots as sm

st.title("Directed Acyclical Graphs (DAGs)")

# Create four buttons in a row
buttons = st.columns(4)

with buttons[0]:
    collider_button = st.button('Generate Collider DAG')
with buttons[1]:
    mediator_button = st.button('Generate Mediator DAG')
with buttons[2]:
    RCT_button = st.button('Generate RCT DAG')
with buttons[3]:
    confounding_button = st.button('Generate Confounding DAG')

# Function to plot with a calculated regression line
def plot_with_regression_line(df, x_col, y_col, title, scatter_color='blue', line_color='black', background_color='#e5e5e5'):
    x = df[x_col]
    y = df[y_col]
    coefficients = np.polyfit(x, y, 1)  # Fit a linear regression model
    regression_line = np.polyval(coefficients, x)  # Calculate the regression line
    
    fig, ax = plt.subplots(figsize=(8, 6))
    fig.patch.set_facecolor(background_color)  # Set the background color

    ax.scatter(df[x_col], df[y_col], alpha=0.5, color=scatter_color)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    
    ax.plot(x, regression_line, color=line_color, linewidth=1, label='Regression Line')
    ax.legend()
    st.pyplot(fig)

# Collider DAG
def simulate_collider_data():
    SIZE = 1000
    X = np.random.normal(size=SIZE)
    Y = np.random.normal(size=SIZE)
    e = np.random.normal(size=SIZE)
    Z = 2*X + 1*Y + e
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df

def plot_collider_dag(df):
    plot_with_regression_line(df, 'X', 'Y', 'Collider DAG', scatter_color='blue', line_color='black')

    # Partial regression with Z as a control variable
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df, ax=ax, obs_labels=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)

if collider_button:
    df = simulate_collider_data()
    plot_collider_dag(df)
    
    st.markdown("**Collider DAG Explanation**:")
    st.write("""
    In this DAG, we have three variables: X, Y, and Z. X and Y are independent variables, and Z is a collider, influenced by both X and Y. This situation represents a collider bias scenario, where the path between X and Y is blocked due to the collider Z. Collider bias can lead to misleading conclusions when analyzing causal relationships.
    """)
    mod = smf.ols(formula='Y ~ X + Z', data=df)
    res = mod.fit()
    st.text(res.summary().as_text())
    print(res.summary())

# Mediator DAG
def simulate_mediator_data():
    SIZE = 1000
    X = np.random.normal(size=SIZE)
    Z = 1.5 * X + np.random.normal(size=SIZE)
    e = np.random.normal(size=SIZE)
    Y = 2 * Z + e
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df

def plot_mediator_dag(df):
    plot_with_regression_line(df, 'X', 'Y', 'Mediator DAG')

    # Partial regression with Z as a control variable
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df, ax=ax, obs_labels=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)

if mediator_button:
    df = simulate_mediator_data()
    plot_mediator_dag(df)

    st.markdown("**Mediator DAG Explanation**:")
    st.write("""
    Here, we have three variables: X, Y, and Z. X directly influences Y through Z, acting as a mediator. X indirectly affects Y, and Z plays a crucial role in transmitting the effect of X to Y. Understanding mediator relationships is essential for dissecting causal pathways.
    """)
    mod = smf.ols(formula='Y ~ X + Z', data=df)
    res = mod.fit()
    st.text(res.summary().as_text())
    print(res.summary())

# RCT DAG
def simulate_RCT_data():
    SIZE = 1000

    # X is randomized treatment, so not influenced by any other variable
    X = np.random.normal(size=SIZE)

    # Z is some covariates
    Z = np.random.normal(size=SIZE)

    # e is the error term
    e = np.random.normal(size=SIZE)

    # Y is influenced by both the treatment X and covariates Z
    Y = 1.5 * X + 2 * Z + e

    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df


def plot_RCT_dag(df):
    plot_with_regression_line(df, 'X', 'Y', 'RCT DAG')

    # Partial regression with Z as a control variable
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df, ax=ax, obs_labels=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)

if RCT_button:
    df = simulate_RCT_data()
    plot_RCT_dag(df)

    st.markdown("**RCT DAG Explanation**:")
    st.write("In this DAG, we observe three variables: X, Y, and Z. Z is the common cause of X and Y. It influences both X and Y independently, representing a RCT structure. Studying RCTs helps us understand how a common cause can impact multiple variables in a causal system.")
    mod = smf.ols(formula='Y ~ X + Z', data=df)
    res = mod.fit()
    st.text(res.summary().as_text())
    print(res.summary())

# Confounding DAG
def simulate_confounding_data():
    SIZE = 1000
    Z = np.random.normal(size=SIZE)
    X = Z * 1.5 + np.random.normal(size=SIZE)
    e = np.random.normal(size=SIZE)
    Y = 2 * Z + X * 1.3 + e
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df

def plot_confounding_dag(df):
    plot_with_regression_line(df, 'X', 'Y', 'Confounding DAG')

    # Partial regression with Z as a control variable
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df, ax=ax, obs_labels=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)

if confounding_button:
    df = simulate_confounding_data()
    plot_confounding_dag(df)

    st.markdown("**Confounding DAG Explanation**:")
    st.write("""
    This DAG involves three variables: X, Y, and Z. Z acts as a common cause of both X and Y, while X directly affects Y as well. This scenario illustrates the concept of confounding, where a third variable (Z) influences both the treatment (X) and the outcome (Y). Understanding confounding is crucial in causal inference.
    """)
    mod = smf.ols(formula='Y ~ X + Z', data=df)
    res = mod.fit()
    st.text(res.summary().as_text())
    print(res.summary())
