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
#with buttons[2]:
    #RCT_button = st.button('Generate RCT DAG')
with buttons[2]:
    confounding_button = st.button('Generate Confounding DAG')


st.title("Research Design") 
buttons_under = st.columns(2)
with buttons_under[0]:
    Dif_button = st.button('Difference in Difference')
    Reg_button = st.button('Regression Discontinuity')


# Function to plot with a calculated regression line
# Function to plot with a calculated regression line
def plot_with_regression_line(df, x_col, y_col, scatter_color='#8bcfbd', line_color='black', background_color='#e5e5e5'):
    x = df[x_col]
    y = df[y_col]
    coefficients = np.polyfit(x, y, 1)  # Fit a linear regression model
    regression_line = np.polyval(coefficients, x)  # Calculate the regression line
    
    fig, ax = plt.subplots(figsize=(9, 6))
    fig.patch.set_facecolor(background_color)  # Set the background color

    ax.scatter(df[x_col], df[y_col], alpha=0.5, color=scatter_color)
    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    
    ax.plot(x, regression_line, color=line_color, linewidth=1, label='Regression Line')
    
    # Set the same limits for both x and y axes
    common_limits = [-5, 5]  # Adjust the limits as needed
    ax.set_xlim(common_limits)
    ax.set_ylim(common_limits)
    
    # Print the equation on the plot
    equation_text = f'Equation: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}'
    ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    ax.legend()
    st.pyplot(fig)


# Collider DAG
def simulate_collider_data():
    SIZE = 1000
    X = np.random.normal(size=SIZE)
    Y = np.random.normal(size=SIZE)
    e = np.random.normal(size=SIZE)
    Z = 2*X + 1*Y + e
    df_collider = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df_collider


# Collider DAG
def plot_collider_dag(df_collider, scatter_color='#8bcfbd', line_color='black', background_color='#e5e5e5'):
    plot_with_regression_line(df_collider, 'X', 'Y', scatter_color=scatter_color, line_color=line_color, background_color=background_color)
    
    # Additional plotting specific to Collider DAG
    fig, ax = plt.subplots(figsize=(9, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df_collider, ax=ax, obs_labels=False)
    fig.patch.set_facecolor(background_color)  # Set the background color
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    
    # Set the same limits for both x and y axes
    common_limits = [-5, 5]  # Adjust the limits as needed
    ax.set_xlim(common_limits)
    ax.set_ylim(common_limits)
    
    # Get coefficients for the regression line specific to this dataset
    coefficients = np.polyfit(df_collider['X'], df_collider['Y'], 1)
    
    # Print the equation on the plot
    equation_text = f'Equation: y = {coefficients[0]:.2f}x + {coefficients[1]:.2f}'
    ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=10, verticalalignment='top')
    
    st.pyplot(fig)







# Mediator DAG
def simulate_mediator_data():
    SIZE = 1000
    X = np.random.normal(size=SIZE)
    Z = 1.5 * X + np.random.normal(size=SIZE)
    e = np.random.normal(size=SIZE)
    Y = 2 * Z + e
    df_mediator = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df_mediator

def plot_mediator_dag(df_mediator, scatter_color='#8bcfbd', line_color='black', background_color='#e5e5e5'):
    plot_with_regression_line(df_mediator, 'X', 'Y', scatter_color=scatter_color, line_color=line_color, background_color=background_color)
    # Additional plotting specific to Mediator DAG
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df_mediator, ax=ax, obs_labels=False)
    fig.patch.set_facecolor(background_color)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
     # Set the same limits for both x and y axes
    common_limits = [-5, 5]  # Adjust the limits as needed
    ax.set_xlim(common_limits)
    ax.set_ylim(common_limits)
    st.pyplot(fig)

# RCT DAG
#def simulate_RCT_data():
    SIZE = 1000
    X = np.random.normal(size=SIZE)
    Z = np.random.normal(size=SIZE)
    e = np.random.normal(size=SIZE)
    Y = 1.5 * X + 2 * Z + e
    df_RCT = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df_RCT

#def plot_RCT_dag(df_RCT, scatter_color='#8bcfbd', line_color='black', background_color='#e5e5e5'):
    plot_with_regression_line(df_RCT, 'X', 'Y', scatter_color=scatter_color, line_color=line_color, background_color=background_color)
    # Additional plotting specific to RCT DAG
    fig, ax = plt.subplots(figsize=(8, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df_RCT, ax=ax, obs_labels=False)
    fig.patch.set_facecolor(background_color)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    st.pyplot(fig)

# Confounding DAG
def simulate_confounding_data():
    SIZE = 1000
    Z = np.random.normal(size=SIZE)
    X = Z * 1.5 + np.random.normal(size=SIZE)
    e = np.random.normal(size=SIZE)
    Y = 2 * Z + X * 1.3 + e
    df_confounding = pd.DataFrame({'X': X, 'Y': Y, 'Z': Z})
    return df_confounding

def plot_confounding_dag(df_confounding, scatter_color='#8bcfbd', line_color='black', background_color='#e5e5e5'):
    plot_with_regression_line(df_confounding, 'X', 'Y', scatter_color=scatter_color, line_color=line_color, background_color=background_color)
    # Additional plotting specific to Confounding DAG
    fig, ax = plt.subplots(figsize=(9, 6))
    sm.plot_partregress(endog='Y', exog_i='X', exog_others=['Z'], data=df_confounding, ax=ax, obs_labels=False)
    fig.patch.set_facecolor(background_color)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
     # Set the same limits for both x and y axes
    common_limits = [-5, 5]  # Adjust the limits as needed
    ax.set_xlim(common_limits)
    ax.set_ylim(common_limits)
    st.pyplot(fig)



# Button checks and corresponding actions
if collider_button:
    df_collider = simulate_collider_data()
    plot_collider_dag(df_collider, scatter_color='#8bcfbd')  # You can change 'red' to any valid color code or name
    
    st.markdown("**Collider DAG Explanation**:")
    st.write("""
    In this DAG, we have three variables: X, Y, and Z. X and Y are independent variables, and Z is a collider, influenced by both X and Y. This situation represents a collider bias scenario, where the path between X and Y is blocked due to the collider Z. Collider bias can lead to misleading conclusions when analyzing causal relationships.
    """)
    mod = smf.ols(formula='Y ~ X + Z', data=df_collider)
    res = mod.fit()
    st.text(res.summary().as_text())
    print(res.summary())




if mediator_button:
    df_mediator = simulate_mediator_data()
    plot_mediator_dag(df_mediator, scatter_color='#8bcfbd')

    st.markdown("**Mediator DAG Explanation**:")
    st.write("""
   Mediator:  

En mediator er en variabel, der formidler eller forklarer sammenhængen mellem den uafhængige variabel (X) og den afhængige variabel (Y). Lad os tage et eksempel: 

Eksempel: Undersøgelse af Sammenhængen mellem Motion (X) og Vægttab (Y) 

Hypotese: Motion påvirker vægttab. 

Faktorer at Overveje: Kostvaner som variabel Z. 

Potentiel Forvirring: Hvis vi ikke tager højde for kostvaner som en mediator, kan vi overse, at en væsentlig del af effekten af motion på vægttab faktisk formidles gennem ændringer i kostvaner. Ignorerer vi dette, kan vi fejlagtigt tilskrive hele effekten til motion alene. 
    """)
    mod = smf.ols(formula='Y ~ X + Z', data=df_mediator)
    res = mod.fit()
    st.text(res.summary().as_text())
    print(res.summary())

#if RCT_button:
    df_RCT = simulate_RCT_data()
    plot_RCT_dag(df_RCT, scatter_color='#8bcfbd')

    st.markdown("**RCT DAG Explanation**:")
    st.write("In this DAG, we observe three variables: X, Y, and Z. Z is the common cause of X and Y. It influences both X and Y independently, representing a RCT structure. Studying RCTs helps us understand how a common cause can impact multiple variables in a causal system.")
    mod = smf.ols(formula='Y ~ X + Z', data=df_RCT)
    res = mod.fit()
    st.text(res.summary().as_text())
    print(res.summary())

if confounding_button:
    df_confounding = simulate_confounding_data()
    plot_confounding_dag(df_confounding, scatter_color='#8bcfbd')

    st.markdown("**Confounding DAG Explanation**:")
    st.write("""
    Confounder: 

En confounder er en variabel, der er relateret både til den uafhængige variabel (X) og den afhængige variabel (Y), hvilket kan forvirre analysen af den sande kausale effekt. Lad os se på et eksempel: 

Eksempel: Undersøgelse af Forholdet mellem Kaffeforbrug (X) og Risiko for Hjertesygdomme (Y) 

Hypotese: Højere kaffeforbrug er forbundet med øget risiko for hjertesygdomme. 

Faktorer at Overveje: Alder som variabel Z. 

Potentiel Forvirring: Alder er relateret både til kaffeforbrug og risiko for hjertesygdomme. Hvis vi ikke kontrollerer for alder som en confounder, kan vi fejlagtigt konkludere, at kaffeforbrug direkte påvirker risikoen for hjertesygdomme, når det i virkeligheden kan være alder, der spiller en rolle. 
    """)
    mod = smf.ols(formula='Y ~ X + Z', data=df_confounding)
    res = mod.fit()
    st.text(res.summary().as_text())
    print(res.summary())
