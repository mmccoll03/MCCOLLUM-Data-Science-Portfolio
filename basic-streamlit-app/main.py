import streamlit as st
import pandas as pd


# Title and app descriptions
st.title("Olympic analytics")
st.markdown("Sort through past Olympic Data; use dropdowns, select boxes, and filters to look at old olympic data")

df = pd.read_csv("basic-streamlit-app/data/olympics.csv")
df = df.drop('id', axis=1)


## Different filters
user_selections = {}

# It doesn't make the most sense to iterate through all columns, like weight or height,
# But this still shows the filtering capability of the app. 
# Loop through column names
for col in df.columns:
    choices = ["None"] + list(df[col].unique())
    user_selections[col] = st.selectbox(f"Select a {col}", choices)


# Filter based on user selection
filtered_df = df
for col, selection in user_selections.items():
    if selection != "None":
        filtered_df = filtered_df[filtered_df[col] == selection]

# Display results
st.write(f"{selection}")
st.dataframe(filtered_df)

### Which countries have the most medals
# Create the DataFrame
df2 = df
df2 = df2.dropna(subset=['medal'])

# Group by 'team' and count the number of medals for each type
# I want to group the df2 dataframe by medal, and create a count of the instances of unique values in the medal column
# unstack turns the data wide, where the unique values in 'medal' become new columns. fill_value puts a 0 in the cells with missing vals.
medal_counts = df2.groupby('team')['medal'].value_counts().unstack(fill_value=0)

# You need to make sure the columns are sorted in a different order, since python shapes the matrix in alphabetical order
# It wouldn't make sense to have Bronze, Gold, then silver. 
medal_counts = medal_counts[['Gold', 'Silver', 'Bronze']]
medal_counts.columns = ['Gold', 'Silver', 'Bronze']

# Calculate the total medals for each country
medal_counts['total'] = medal_counts['Gold'] + medal_counts['Silver'] + medal_counts['Bronze']

# Display sliders for total medals and individual medal types
min_total = medal_counts['total'].min()
max_total = medal_counts['total'].max()

# Sliders for filtering
master_slide = st.selectbox("See which countries have the most medals:", ["Total medals", "Gold", "Silver", "Bronze"])

# Only show sliders and filtering if "Filter by medals" is selected
if master_slide == "Total medals":
    # Get the min and max values for the sliders
    min_total = medal_counts['total'].min()
    max_total = medal_counts['total'].max()
    total_slider = st.slider("Select Total Medals", min_value=min_total, max_value=max_total, value=(min_total, max_total))

    filtered_df2 = medal_counts[
        (medal_counts['total'] >= total_slider[0]) & 
        (medal_counts['total'] <= total_slider[1]) 
    ]

elif master_slide == "Gold":
    gold_slider = st.slider("Select Gold Medals", min_value=0, max_value=medal_counts['Gold'].max(), value=(0, medal_counts['Gold'].max()))
    filtered_df2 = medal_counts[
        (medal_counts['Gold'] >= gold_slider[0]) & 
        (medal_counts['Gold'] <= gold_slider[1])
    ]
elif master_slide == "Silver":
    silver_slider = st.slider("Select Silver Medals", min_value=0, max_value=medal_counts['Silver'].max(), value=(0, medal_counts['Silver'].max()))
    filtered_df2 = medal_counts[
        (medal_counts['Silver'] >= silver_slider[0]) & 
        (medal_counts['Silver'] <= silver_slider[1])
    ]
else:
    bronze_slider = st.slider("Select Bronze Medals", min_value=0, max_value=medal_counts['Bronze'].max(), value=(0, medal_counts['Bronze'].max()))
    filtered_df2 = medal_counts[
            (medal_counts['Bronze'] >= bronze_slider[0]) & 
            (medal_counts['Bronze'] <= bronze_slider[1])
    ]


# Display the filtered results
st.write(filtered_df2)


#SANITY CHECK
# st.write("Break")
# st.write(df2[(df2['team'] == 'United States')].shape[0])
# st.write(df[df['team'] == 'United States']['medal'].isna().sum())
# st.write(df[(df['team'] == 'United States')].shape[0])

import matplotlib.pyplot as plt
import seaborn as sns


# It would be nice, in the future, to make this an interactive graph, so one could click on a country to see the athletes with the most medals
# or see a time series of medals won. But I didn't really have time to implement this. 
st.write("Total Medals by Country")
total_medals_plot = medal_counts['total'].sort_values(ascending=False).head(10)  # Top 10 countries
total_medals_plot.plot(kind='bar', title="Top 10 Countries by Total Medals", figsize=(10, 6))
plt.xlabel('Country')
plt.ylabel('Total Medals')
st.pyplot(plt)


## little section about the US team
st.header("US Olympic Athletes")

st.image('basic-streamlit-app/usa.png')

df3 = df[df['team'] == 'United States']
df3 = df3.drop_duplicates(subset='name')

sports = df3['sport'].unique()
options = ['Male', 'Female', 'Summer', 'Winter'] + list(sports)
selection = st.multiselect('Multiselect', options)

#filter
if 'Male' in selection:
    df3 = df3[df3['sex'] == 'M']
elif 'Female' in selection:
    df3 = df3[df3['sex'] == 'F']

if 'Summer' in selection:
    df3 = df3[df3['season'] == 'Summer']
elif 'Winter' in selection:
    df3 = df3[df3['season'] == 'Winter']

selected_sports = [sport for sport in selection if sport in sports]
if selected_sports:
    df3 = df3[df3['sport'].isin(selected_sports)]

st.write(df3)