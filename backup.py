import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import statsmodels.graphics.regressionplots as sm

st.title("Directed Acyclical Graphs (DAGs)")

# Collider DAG
def simulate_collider_data():
    SIZE = 1000
    X = np.random.normal(size=SIZE)
    Y = np.random.normal(size=SIZE)
    e = np.random.normal(size=SIZE)  # noise
    Z = 2*X + 1*Y + e
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df

def plot_collider_dag(df):
    # Y -> Z <- X
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=[], data=df, ax=ax, obs_labels=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)  # Explicitly pass the figure object to Streamlit
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df, ax=ax, obs_labels=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)


# Button for Collider DAG
if st.button('Generate Collider DAG'):
    df = simulate_collider_data()
    plot_collider_dag(df)

# ... (other DAGs can follow a similar pattern)
# Mediator DAG
def simulate_mediator_data():
    SIZE = 1000
    X = np.random.normal(size=SIZE)
    Z = 1.5 * X + np.random.normal(size=SIZE)
    e = np.random.normal(size=SIZE)  # noise
    Y = 2 * Z + e
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df

def plot_mediator_dag(df):
    # X -> Z -> Y
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=[], data=df, ax=ax, obs_labels=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)  # Explicitly pass the figure object to Streamlit
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df, ax=ax, obs_labels=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)


# Button for Mediator DAG
if st.button('Generate Mediator DAG'):
    df = simulate_mediator_data()
    plot_mediator_dag(df)

# Fork DAG
def simulate_fork_data():
    SIZE = 1000
    Z = np.random.normal(size=SIZE)
    X = Z * 1.5 + np.random.normal(size=SIZE)
    e = np.random.normal(size=SIZE)  # noise
    Y = 2 * Z + e
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df

def plot_fork_dag(df):
    # X <- Z -> Y
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=[], data=df, ax=ax, obs_labels=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)  # Explicitly pass the figure object to Streamlit
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df, ax=ax, obs_labels=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)

# Button for Fork DAG
if st.button('Generate Fork DAG'):
    df = simulate_fork_data()
    plot_fork_dag(df)

# Confounding DAG
def simulate_confounding_data():
    SIZE = 1000
    Z = np.random.normal(size=SIZE)
    X = Z * 1.5 + np.random.normal(size=SIZE)
    e = np.random.normal(size=SIZE)  # noise
    Y = 2 * Z + X * 1.3 + e
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df

def plot_confounding_dag(df):
    # X <- Z -> Y, X -> Y
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=[], data=df, ax=ax, obs_labels=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)  # Explicitly pass the figure object to Streamlit
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df, ax=ax, obs_labels=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)

# Button for Confounding DAG
if st.button('Generate Confounding DAG'):
    df = simulate_confounding_data()
    plot_confounding_dag(df)

# ... [any other DAGs or Streamlit components you have]



