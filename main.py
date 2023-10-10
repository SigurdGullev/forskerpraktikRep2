import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
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
    fork_button = st.button('Generate Fork DAG')
with buttons[3]:
    confounding_button = st.button('Generate Confounding DAG')

# Collider DAG
def simulate_collider_data():
    SIZE = 1000
    X = np.random.uniform(0, 10, SIZE)  # values between 0 and 10
    Y = np.random.uniform(0, 10, SIZE)  # values between 0 and 10
    e = np.random.normal(size=SIZE)     # noise
    Z = 2*X + 1 *Y + e
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df


# bliver man idiot af at være smuk, x- aksen er skønhed, y-akse er intelligens
def plot_collider_dag(df):
    # Y -> Z <- X
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=[], data=df, ax=ax, obs_labels=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    st.pyplot(fig)  
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df, ax=ax, obs_labels=False)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    st.pyplot(fig)



# Collider DAG
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


# Mediator DAG
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

# Fork DAG
if fork_button:
    df = simulate_fork_data()
    plot_fork_dag(df)
    st.markdown("**Fork DAG Explanation**:")
    st.write("""
    In this DAG, we observe three variables: X, Y, and Z. Z is the common cause of X and Y. It influences both X and Y independently, representing a fork structure. Studying forks helps us understand how a common cause can impact multiple variables in a causal system.
    """)
    mod = smf.ols(formula='Y ~ X + Z', data=df)
    res = mod.fit()
    st.text(res.summary().as_text())
    print(res.summary())



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

# Confounding DAG
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

