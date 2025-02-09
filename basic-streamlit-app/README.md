# This is a basic Streamlit app which reads data and has interactivity widgets in Streamlit.

## Data Includes:
A csv file from tidy Tuesday that has around 271,000 rows of Olympic athletes from the the 1900s and 2000s. The variables include their names, their sex, age, height, weight, team, which games they played, what year they competed, which season it was, the city they competed in, the sport and event they competed in, and if they won a medal. 

This app involves some filtering functions to look at the data based on medals won, and to look at athletes that were on the US team. 

Note: While the code seems to work correctly, I don't think this data itself is competely accurate. There are some strange team names, and the medals counts don't add up to numbers that make sense. After a lot of troubleshooting, I think the medals counts are true for what is in the data, but these numbers do not reflect true medal counts. I don't know what the explanation for this would be, but there is still basic functionality within the app. 

## Use "streamlit run basic_streamlit_app/main.py" to open the app

The data is in the data folder as olympics.csv