import streamlit as st
import pandas as pd

# Read CSV file
df = pd.read_csv('mutant_moneyball.csv')

# Clean data
df = df.fillna(0)

# Clean up the dollar signs and commas for monetary values
cols = [
    'TotalValue60s_wiz', 'TotalValue70s_wiz', 'TotalValue80s_wiz', 'TotalValue90s_wiz',
    'TotalValue60s_oStreet', 'TotalValue70s_oStreet', 'TotalValue80s_oStreet', 'TotalValue90s_oStreet'
]

for col in cols:
    df[col] = df[col].astype(str).str.replace(r'[\$,]', '', regex=True).str.strip().astype(float)

# Melt the DataFrame to long format
df_melted = df.melt(id_vars=['Member'], var_name='Decade_Market', value_name='TotalValue')

# Split 'Decade_Market' into 'Decade' and 'Market'
df_melted[['Decade', 'Market']] = df_melted['Decade_Market'].str.extract(r'TotalValue(?P<Decade>\d{2}s)_(?P<Market>\w+)')

# Drop 'Decade_Market' column
df_melted = df_melted.drop(columns=['Decade_Market'])

st.set_page_config(page_title="Mutant Data Dashboard", layout="wide")
st.title("Mutant Value Analysis")

# Multiselect box for selecting mutants
mutants_selected = st.multiselect("Select Mutants", df['Member'].unique())


col1, col2, col3 = st.columns([1, 2, 1])

# Decades as horizontal buttons
with col1:
    decade_buttons = ['60s', '70s', '80s', '90s', 'All Time']
    decade_selected = st.radio("Select Decade", decade_buttons, horizontal=True)


with col3:
    market_selected = st.selectbox("Select Market", ['All Markets', 'ebay', 'heritage', 'oStreet', 'wiz'])

st.write(f"Selected Mutants: {mutants_selected}")
st.write(f"Selected Decade: {decade_selected}")
st.write(f"Selected Market: {market_selected}")

# Filter the data based on selected values
df_filtered = df_melted[df_melted['Member'].isin(mutants_selected)]

# Filter by Decade
if decade_selected != 'All Time':
    df_filtered = df_filtered[df_filtered['Decade'] == decade_selected]

# Filter by Market
if market_selected != 'All Markets':
    df_filtered = df_filtered[df_filtered['Market'] == market_selected]

# Display the filtered dataframe
st.write(f"### Filtered Data")
st.dataframe(df_filtered)
