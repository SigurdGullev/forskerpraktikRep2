import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import statsmodels.formula.api as smf
import statsmodels.graphics.api as sm

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
