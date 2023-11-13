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
    
    st.markdown("**Collider DAG Forklaring**:")
    st.write("""
   Collider bias er en forvrængning, der ændrer en sammenhæng mellem en eksponering og et resultat, forårsaget af forsøg på at kontrollere for en fælles effekt.

**Eksempel:** Undersøgelse af Sammenhængen mellem Locomotor Sygdom (X) og Respiratorisk Sygdom (Y) 

**Hypotese:** Der antages en sammenhæng mellem locomotor sygdom og respiratorisk sygdom.

**Faktorer at Overveje:** En vigtig faktor at overveje er, om der er en fælles variabel, såsom en collider, der formidler eller forvrænger sammenhængen. I Sacketts eksempel fra 1979 analyserede han data fra 257 indlagte personer og påviste en sammenhæng mellem locomotor sygdom og respiratorisk sygdom (oddsforhold 4.06). Collideren i dette tilfælde er indlæggelse. Indlæggelse fungerer som en collider, fordi både locomotor sygdom og respiratorisk sygdom kan føre til indlæggelse. Når vi forsøger at kontrollere for indlæggelse (fx ved studiedesign eller statistisk analyse), skaber det collider bias. Det skaber en forvrænget sammenhæng mellem locomotor sygdom og respiratorisk sygdom, selvom der måske ikke er nogen reel årsagssammenhæng mellem de to, da begge kan føre til indlæggelse.


**Potentiel Forvirring:**    Hvis vi ikke tager højde for indlæggelse som en mulig collider, kan vi overse, at en væsentlig del af effekten af locomotor sygdom på respiratorisk sygdom faktisk formidles gennem indlæggelse. Ignorerer vi dette, kan vi fejlagtigt tilskrive hele sammenhængen mellem locomotor sygdom og respiratorisk sygdom til locomotor sygdom alene.
""")
    mod = smf.ols(formula='Y ~ X + Z', data=df_collider)
    res = mod.fit()
    st.text(res.summary().as_text())
    print(res.summary())




if mediator_button:
    df_mediator = simulate_mediator_data()
    plot_mediator_dag(df_mediator, scatter_color='#8bcfbd')
    # Display an image from a local file
    image_path = "/Users/sigurdgullev/Repositories/forskerpraktikRep2/images/Screenshot 2023-11-13 at 09.54.33.png"
    st.image(image_path, caption='', use_column_width=True)
    st.markdown("**Mediator DAG Forklaring**:")
    st.write("""
En mediator er en variabel, der formidler eller forklarer sammenhængen mellem den uafhængige variabel (X) og den afhængige variabel (Y). Lad os tage et eksempel: 

**Eksempel:** Undersøgelse af Sammenhængen mellem Motion (X) og Vægttab (Y) 

**Hypotese:** Motion påvirker vægttab. 

**Faktorer at Overveje:** Kostvaner som variabel Z. 

**Potentiel Forvirring:** Hvis vi ikke tager højde for kostvaner som en mediator, kan vi overse, at en væsentlig del af effekten af motion på vægttab faktisk formidles gennem ændringer i kostvaner. Ignorerer vi dette, kan vi fejlagtigt tilskrive hele effekten til motion alene. 
    """)
    mod = smf.ols(formula='Y ~ X + Z', data=df_mediator)
    res = mod.fit()
    st.text(res.summary().as_text())
    print(res.summary())


if confounding_button:
    df_confounding = simulate_confounding_data()
    plot_confounding_dag(df_confounding, scatter_color='#8bcfbd')
     # Display an image from a local file
    image_path = "/Users/sigurdgullev/Repositories/forskerpraktikRep2/images/Screenshot 2023-11-13 at 09.57.37.png"
    st.image(image_path, caption='', use_column_width=True)
    st.markdown("**Confounding DAG Forklaring**:")
    st.write("""
En confounder er en variabel, der er relateret både til den uafhængige variabel (X) og den afhængige variabel (Y), hvilket kan forvirre analysen af den sande kausale effekt. Lad os se på et eksempel: 

**Eksempel:** Undersøgelse af Forholdet mellem Kaffeforbrug (X) og Risiko for Hjertesygdomme (Y) 

**Hypotese:** Højere kaffeforbrug er forbundet med øget risiko for hjertesygdomme. 

**Faktorer at Overveje:** Alder som variabel Z. 

**Potentiel Forvirring:** Alder er relateret både til kaffeforbrug og risiko for hjertesygdomme. Hvis vi ikke kontrollerer for alder som en confounder, kan vi fejlagtigt konkludere, at kaffeforbrug direkte påvirker risikoen for hjertesygdomme, når det i virkeligheden kan være alder, der spiller en rolle. 
    """)
    mod = smf.ols(formula='Y ~ X + Z', data=df_confounding)
    res = mod.fit()
    st.text(res.summary().as_text())
    print(res.summary())
