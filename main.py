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

# Collider DAG: Exercise, Diet Quality, and Weight
def simulate_collider_data():
    SIZE = 1000
    
    # Using normal distribution for a smoother distribution of data points
    Exercise = np.round(np.random.normal(2, 1, SIZE))  # center around 2 hours with std deviation of 1 hour
    Diet_Quality = np.round(np.random.normal(5.5, 2, SIZE))  # center around 5.5 with std deviation of 2
    
    # Ensure values are within desired bounds
    Exercise = np.clip(Exercise, 0, 4)
    Diet_Quality = np.clip(Diet_Quality, 1, 10)
    
    e = np.random.normal(size=SIZE)  # noise
    Weight = 80 - 2*Exercise - 1*Diet_Quality + e  # 80 is an average weight for the population
    df = pd.DataFrame({'Exercise': Exercise, 'Diet_Quality': Diet_Quality, 'Weight': Weight})
    return df


def plot_collider_dag(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Weight', exog_i='Exercise', exog_others=[], data=df, ax=ax, obs_labels=False)
    st.pyplot(fig)  
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Weight', exog_i='Exercise', exog_others=['Diet_Quality'], data=df, ax=ax, obs_labels=False)
    st.pyplot(fig)

if collider_button:
    df = simulate_collider_data()
    plot_collider_dag(df)
    st.markdown("**Collider DAG Explanation**:")
    st.write("In this DAG, we have three variables: Exercise, Diet Quality, and Weight. When people exercise more and maintain a better diet, they tend to weigh less. However, there could be third variables, like metabolism, that can confound these associations.")
    mod = smf.ols(formula='Weight ~ Exercise + Diet_Quality', data=df)
    res = mod.fit()
    st.text(res.summary().as_text())

# Mediator DAG: Education, Job Skill Level, and Income
def simulate_mediator_data():
    SIZE = 1000
    Education = np.random.randint(12, 21, SIZE)  # 12-20 years of education
    Job_Skill_Level = Education + np.random.normal(0, 2, SIZE)
    e = np.random.normal(size=SIZE)
    Income = 20000 + 5000 * Job_Skill_Level + e
    df = pd.DataFrame({'Education': Education, 'Job_Skill_Level': Job_Skill_Level, 'Income': Income})
    return df

def plot_mediator_dag(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Income', exog_i='Education', exog_others=[], data=df, ax=ax, obs_labels=False)
    st.pyplot(fig)  
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Income', exog_i='Education', exog_others=['Job_Skill_Level'], data=df, ax=ax, obs_labels=False)
    st.pyplot(fig)

if mediator_button:
    df = simulate_mediator_data()
    plot_mediator_dag(df)
    st.markdown("**Mediator DAG Explanation**:")
    st.write("In this DAG, we explore the relationship between Education, Job Skill Level, and Income. Typically, as people achieve higher education, they acquire more specialized job skills, which in turn leads to a higher income.")
    mod = smf.ols(formula='Income ~ Education + Job_Skill_Level', data=df)
    res = mod.fit()
    st.text(res.summary().as_text())

# Fork DAG: Parental Education, Household Income, and Child's Health
def simulate_fork_data():
    SIZE = 1000
    Parental_Education = np.random.randint(10, 21, SIZE)
    Household_Income = 20000 + 3000 * Parental_Education
    e = np.random.normal(size=SIZE)
    Child_Health = 90 + 0.05 * Household_Income + e  # Assuming a scale of 0-100 for health
    df = pd.DataFrame({'Parental_Education': Parental_Education, 'Household_Income': Household_Income, 'Child_Health': Child_Health})
    return df

def plot_fork_dag(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Child_Health', exog_i='Parental_Education', exog_others=[], data=df, ax=ax, obs_labels=False)
    st.pyplot(fig)  
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Child_Health', exog_i='Parental_Education', exog_others=['Household_Income'], data=df, ax=ax, obs_labels=False)
    st.pyplot(fig)

if fork_button:
    df = simulate_fork_data()
    plot_fork_dag(df)
    st.markdown("**Fork DAG Explanation**:")
    st.write("In this DAG, we consider the impact of Parental Education on Household Income and Child's Health. Generally, higher parental education can lead to a higher household income, which in turn can positively affect a child's health.")
    mod = smf.ols(formula='Child_Health ~ Parental_Education + Household_Income', data=df)
    res = mod.fit()
    st.text(res.summary().as_text())

# Confounding DAG: Outdoor Activity, Sun Exposure, and Vitamin D Level
def simulate_confounding_data():
    SIZE = 1000
    Outdoor_Activity = np.random.randint(1, 6, SIZE)  # 1-5 hours per week
    Sun_Exposure = Outdoor_Activity + np.random.normal(0, 1, SIZE)
    e = np.random.normal(size=SIZE)
    Vitamin_D_Level = 20 + 5 * Sun_Exposure + e
    df = pd.DataFrame({'Outdoor_Activity': Outdoor_Activity, 'Sun_Exposure': Sun_Exposure, 'Vitamin_D_Level': Vitamin_D_Level})
    return df

def plot_confounding_dag(df):
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Vitamin_D_Level', exog_i='Outdoor_Activity', exog_others=[], data=df, ax=ax, obs_labels=False)
    st.pyplot(fig)  
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Vitamin_D_Level', exog_i='Outdoor_Activity', exog_others=['Sun_Exposure'], data=df, ax=ax, obs_labels=False)
    st.pyplot(fig)

if confounding_button:
    df = simulate_confounding_data()
    plot_confounding_dag(df)
    st.markdown("**Confounding DAG Explanation**:")
    st.write("""
    In this DAG, we delve into the relationship between Outdoor Activity, Sun Exposure, and Vitamin D Level. People who engage in more outdoor activities generally get more sun exposure, leading to higher Vitamin D levels. However, Sun Exposure acts as a confounder here, affecting both the Outdoor Activity and Vitamin D Level directly.
    """)
    mod = smf.ols(formula='Vitamin_D_Level ~ Outdoor_Activity + Sun_Exposure', data=df)
    res = mod.fit()
    st.text(res.summary().as_text())