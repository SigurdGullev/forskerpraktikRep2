import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import statsmodels.graphics.regressionplots as sm

st.title("Directed Acyclical Graphs (DAGs)")
col1, col2, col3, col4 = st.columns(4)
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
with col1:
    if st.button('Generate Collider DAG'):
        df = simulate_collider_data()
        plot_collider_dag(df)
        st.write("""
        **Collider DAG Explanation**:
        
        In this DAG, we have three variables: X, Y, and Z. X and Y are independent variables, and Z is a collider, influenced by both X and Y. This situation represents a collider bias scenario, where the path between X and Y is blocked due to the collider Z. Collider bias can lead to misleading conclusions when analyzing causal relationships.
        """)

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
with col2:
    
    if st.button('Generate Mediator DAG'):
        df = simulate_mediator_data()
        plot_mediator_dag(df)
        st.write("""
        **Mediator DAG Explanation**:
        
        Here, we have three variables: X, Y, and Z. X directly influences Y through Z, acting as a mediator. X indirectly affects Y, and Z plays a crucial role in transmitting the effect of X to Y. Understanding mediator relationships is essential for dissecting causal pathways.
        """)

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
with col3:
    if st.button('Generate Fork DAG'):
        df = simulate_fork_data()
        plot_fork_dag(df)
        st.write("""
        **Fork DAG Explanation**:
        
        In this DAG, we observe three variables: X, Y, and Z. Z is the common cause of X and Y. It influences both X and Y independently, representing a fork structure. Studying forks helps us understand how a common cause can impact multiple variables in a causal system.
        """)

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
with col4:
    if st.button('Generate Confounding DAG'):
        df = simulate_confounding_data()
        plot_confounding_dag(df)
        st.write("""
        **Confounding DAG Explanation**:
        
        This DAG involves three variables: X, Y, and Z. Z acts as a common cause of both X and Y, while X directly affects Y as well. This scenario illustrates the concept of confounding, where a third variable (Z) influences both the treatment (X) and the outcome (Y). Understanding confounding is crucial in causal inference.
        """)
