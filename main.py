import streamlit as st
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
import statsmodels.api as sm

st.title("Directed Acyclic Graphs (DAGs)")

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

# Set common plot settings
def set_plot_settings():
    fig, ax = plt.subplots()
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    return fig, ax

# Collider DAG
def simulate_collider_data():
    SIZE = 1000
    X = np.random.uniform(0, 10, size=SIZE)
    Y = np.random.uniform(0, 10, size=SIZE)
    e = np.random.normal(size=SIZE)
    Z = 2 * X + 1 * Y + e
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df

def plot_collider_dag(df):
    fig, ax = set_plot_settings()
    ax.scatter(df['X'], df['Y'], alpha=0.5)
    st.pyplot(fig)

    fig, ax = set_plot_settings()
    sm.graphics.regressionplots.plot_partregress('Y', 'X', exog_others=['Z'], data=df, obs_labels=False, ax=ax)
    st.pyplot(fig)

if collider_button:
    df = simulate_collider_data()
    plot_collider_dag(df)
    
    st.markdown("**Collider DAG Explanation**:")
    st.write("""
    In this Collider DAG, we have three variables: X, Y, and Z. X and Y are independent variables, and Z is a collider, influenced by both X and Y. This situation represents a collider bias scenario, where the path between X and Y is blocked due to the collider Z. Collider bias can lead to misleading conclusions when analyzing causal relationships. In this example, X and Y are not directly related, but their relationship is influenced by the collider Z.
    """)
    mod = smf.ols('Y ~ X + Z', data=df)
    res = mod.fit()
    st.text("Regression Summary:")
    st.text(res.summary().as_text())

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
    fig, ax = set_plot_settings()
    ax.scatter(df['X'], df['Y'], alpha=0.5)
    st.pyplot(fig)

    fig, ax = set_plot_settings()
    sm.graphics.regressionplots.plot_partregress('Y', 'X', exog_others=['Z'], data=df, obs_labels=False, ax=ax)
    st.pyplot(fig)

if mediator_button:
    df = simulate_mediator_data()
    plot_mediator_dag(df)

    st.markdown("**Mediator DAG Explanation**:")
    st.write("""
    In this Mediator DAG, we have three variables: X, Y, and Z. X directly influences Y through Z, acting as a mediator. X indirectly affects Y, and Z plays a crucial role in transmitting the effect of X to Y. Understanding mediator relationships is essential for dissecting causal pathways. In this example, X has an indirect effect on Y through the mediator Z, and it's important to control for Z when analyzing the relationship between X and Y.
    """)
    mod = smf.ols('Y ~ X + Z', data=df)
    res = mod.fit()
    st.text("Regression Summary:")
    st.text(res.summary().as_text())

# RCT DAG
def simulate_RCT_data():
    SIZE = 1000
    X = np.random.normal(size=SIZE)
    Z = np.random.normal(size=SIZE)
    e = np.random.normal(size=SIZE)
    Y = 1.5 * X + 2 * Z + e
    df = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df

def plot_RCT_dag(df):
    fig, ax = set_plot_settings()
    ax.scatter(df['X'], df['Y'], alpha=0.5)
    st.pyplot(fig)

    fig, ax = set_plot_settings()
    sm.graphics.regressionplots.plot_partregress('Y', 'X', exog_others=['Z'], data=df, obs_labels=False, ax=ax)
    st.pyplot(fig)

if RCT_button:
    df = simulate_RCT_data()
    plot_RCT_dag(df)

    st.markdown("**RCT DAG Explanation**:")
    st.write("In this RCT (Randomized Control Trial) DAG, we observe three variables: X, Y, and Z. Z is the common cause of X and Y. It influences both X and Y independently, representing a RCT structure. Studying RCTs helps us understand how a common cause can impact multiple variables in a causal system.")
    mod = smf.ols('Y ~ X + Z', data=df)
    res = mod.fit()
    st.text("Regression Summary:")
    st.text(res.summary().as_text())

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
    fig, ax = set_plot_settings()
    ax.scatter(df['X'], df['Y'], alpha=0.5)
    st.pyplot(fig)

    fig, ax = set_plot_settings()
    sm.graphics.regressionplots.plot_partregress('Y'),
