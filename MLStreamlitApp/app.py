import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from pages import introduction, linear_regression, logistic_regression, knn, decision_tree

# Set page configuration
st.set_page_config(page_title="Machine Learning App", layout="wide")

# Global CSS styling for dark theme and Garamond font
st.markdown("""
    <style>
        /* Overall dark background and light text with Garamond font */
        body {
            background-color: #121212;
            color: #f0f0f0;
            font-family: Garamond, serif;
        }
        /* Ensure the Streamlit app container uses the dark theme */
        .stApp {
            background-color: #121212;
        }
        /* Style the sidebar with a slightly different dark tone */
        .stSidebar {
            background-color: #1e1e1e;
        }
        /* Style buttons for a modern look */
        .stButton > button {
            background-color: #8a2be2;
            color: #ffffff;
        }
        .stButton > button:hover {
            background-color: #9b30ff;
        }
        /* Ensure all headings and paragraphs use light text */
        h1, h2, h3 {
            color: #f0f0f0;
        }
        p, li {
            color: #f0f0f0;
        }
        /* Optionally hide default navigation elements */
        [data-testid="stSidebarNav"], header {
            display: none;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar title and navigation buttons
st.sidebar.title("Chapters")

# Dictionary mapping page names to modules
pages = {
    "Introduction": introduction,
    "Linear Regression": linear_regression,
    "Logistic Regression on a Perceptron": logistic_regression,
    "KNN": knn,
    "Decision Tree": decision_tree
}

# Initialize session state for current page if not set
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Introduction"

# Create sidebar buttons for navigation
for page_name in pages:
    if st.sidebar.button(page_name, key=page_name):
        st.session_state.current_page = page_name

# Load and display the currently selected page
current_page_module = pages[st.session_state.current_page]
current_page_module.show()
