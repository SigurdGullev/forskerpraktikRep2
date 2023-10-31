import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.formula.api as smf
import statsmodels.graphics.regressionplots as sm

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
    plt.figure(figsize=(8, 6))
    plt.xticks([])
    plt.yticks([])
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.gca().spines['bottom'].set_visible(False)
    plt.gca().spines['left'].set_visible(False)

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
    set_plot_settings()
    plt.scatter(df['X'], df['Y'], alpha=0.5)
    st.pyplot()

    set_plot_settings()
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df, obs_labels=False)
    st.pyplot()

if collider_button:
    df = simulate_collider_data()
    plot_collider_dag(df)
    
    st.markdown("**Collider DAG Explanation**:")
    st.write("""
    In this Collider DAG, we have three variables: X, Y, and Z. X and Y are independent variables, and Z is a collider, influenced by both X and Y. This situation represents a collider bias scenario, where the path between X and Y is blocked due to the collider Z. Collider bias can lead to misleading conclusions when analyzing causal relationships. In this example, X and Y are not directly related, but their relationship is influenced by the collider Z.
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
    set_plot_settings()
    plt.scatter(df['X'], df['Y'], alpha=0.5)
    st.pyplot()

    set_plot_settings()
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df, obs_labels=False)
    st.pyplot()

if mediator_button:
    df = simulate_mediator_data()
    plot_mediator_dag(df)

    st.markdown("**Mediator DAG Explanation**:")
    st.write("""
    In this Mediator DAG, we have three variables: X, Y, and Z. X directly influences Y through Z, acting as a mediator. X indirectly affects Y, and Z plays a crucial role in transmitting the effect of X to Y. Understanding mediator relationships is essential for dissecting causal pathways. In this example, X has an indirect effect on Y through the mediator Z, and it's important to control for Z when analyzing the relationship between X and Y.
    """)
    mod = smf.ols(formula='Y ~ X + Z', data=df)
    res = mod.fit()
    st.text(res.summary().as_text())
    print(res.summary())

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
    set_plot_settings()
    plt.scatter(df['X'], df['Y'], alpha=0.5)
    st.pyplot()

    set_plot_settings()
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df, obs_labels=False)
    st.pyplot()

if RCT_button:
    df = simulate_RCT_data()
    plot_RCT_dag(df)

    st.markdown("**RCT DAG Explanation**:")
    st.write("In this RCT (Randomized Control Trial) DAG, we observe three variables: X, Y, and Z. Z is the common cause of X and Y. It influences both X and Y independently, representing a RCT structure. Studying RCTs helps us understand how a common cause can impact multiple variables in a causal system. In this example, X is a randomized treatment, and Y is the outcome. Z acts as a covariate that is not influenced by the treatment X.")
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
    set_plot_settings()
    plt.scatter(df['X'], df['Y'], alpha=0.5)
    st.pyplot()

    set_plot_settings()
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df, obs_labels=False)
    st.pyplot()

if confounding_button:
    df = simulate_confounding_data()
    plot_confounding_dag(df)

    st.markdown("**Confounding DAG Explanation**:")
    st.write("""
    This Confounding DAG involves three variables: X, Y, and Z. Z acts as a common cause of both X and Y, while X directly affects Y as well. This scenario illustrates the concept of confounding, where a third variable (Z) influences both the treatment (X) and the outcome (Y). Understanding confounding is crucial in causal inference. In this example, X has a direct effect on Y, but it's also influenced by the common cause Z, leading to potential confounding in the analysis. It's essential to control for Z when studying the relationship between X and Y.
    """)
    mod = smf.ols(formula='Y ~ X + Z', data=df)
    res = mod.fit()
    st.text(res.summary().as_text())
    print(res.summary())
