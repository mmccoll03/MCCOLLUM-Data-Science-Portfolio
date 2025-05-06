# Basic Streamlit Olympic Dashboard

## Project Overview

**Basic Streamlit Olympic Dashboard** is an interactive app built with Python and Streamlit to explore historical Olympic athlete data from the Tidy Tuesday repository. It allows users to filter and visualize athlete statistics—such as age, height, weight, team, and medal counts—across different years and seasons.

## Instructions

### Running the App Locally

1. **Clone the Repository**  
   git clone https://github.com/mmccoll03/MCCOLLUM-Data-Science-Portfolio.git  
   cd MCCOLLUM-Data-Science-Portfolio/basic-streamlit-app

2. **Set Up a Virtual Environment**  
   python -m venv venv  
   source venv/bin/activate    # On Windows: venv\Scripts\activate

3. **Install Dependencies**  
   Ensure your requirements.txt contains at least:  
   pandas  
   streamlit  
   matplotlib  
   seaborn  
   Then run:  
   pip install -r requirements.txt

4. **Launch the App**  
   streamlit run main.py  
   Open your browser to http://localhost:8501.

## App Features

- **Data Filtering**  
  - Filter athletes by medal status (Gold, Silver, Bronze, None).  
  - Filter by team (e.g., United States) and season (Summer/Winter).  
  - Select specific years or year ranges to narrow the dataset.

- **Interactive Visualizations**  
  - Dynamic histograms and bar charts showing age, height, weight distributions.  
  - Time-series plots of medal counts by year or team.  
  - Summary panels with key statistics that update as filters change.

- **Data Quality Notice**  
  The source dataset may contain inconsistencies—unusual team names or medal count mismatches—that reflect issues in the raw data but do not affect the app’s interactive functionality.

## Data

All data is stored in data/olympics.csv, containing ~271,000 records from the 1900s–2000s with the following fields:  
- Name, Sex, Age, Height, Weight  
- Team, Games, Year, Season, City  
- Sport, Event, Medal

## References

- Tidy Tuesday – Olympics Data: https://github.com/rfordatascience/tidytuesday  
- Streamlit Documentation: https://docs.streamlit.io/  
- Pandas Documentation: https://pandas.pydata.org/
