import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# import your page modules from renamed folder
from app_pages import introduction, pca, kmeans_clustering, hierarchical_clustering

# Page configuration
st.set_page_config(page_title="Unsupervised ML App", layout="wide")

# Sidebar navigation
st.sidebar.title("Chapters")

pages = {
    "Introduction": introduction,
    "Principal Component Analysis": pca,
    "K-Means Clustering": kmeans_clustering,
    "Hierarchical Clustering": hierarchical_clustering
}

# Initialize session state
if 'current_page' not in st.session_state:
    st.session_state.current_page = "Introduction"

# Sidebar buttons
for page_name, page_module in pages.items():
    if st.sidebar.button(page_name):
        st.session_state.current_page = page_name

# Display selected page
current_page = pages[st.session_state.current_page]
current_page.show()
