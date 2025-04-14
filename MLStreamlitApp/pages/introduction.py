import streamlit as st

def show():
    st.markdown("""
    <div style="text-align: center; font-family: Garamond, serif;">
        <h1 style="color: #d9a4ff;">Supervised Machine Learning Portfolio</h1>
        <p style="font-size: 18px; color: #f6eaff;">
            Explore various techniques for learning from labeled data.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("""
    **Supervised Machine Learning** is a branch of machine learning 
    where models are trained on labeled data. By providing a dataset 
    with known outcomes, the algorithms learn to make predictions, classify data, 
    or estimate future values. In this portfolio, you will find interactive
    examples that demonstrate several powerful methods:
    
    - **Linear Regression:** Predicts values by fitting a line that minimizes error.
    - **Logistic Regression on a Perceptron:** Classifies data into binary categories using a logistic function.
    - **K-Nearest Neighbors (KNN):** Classifies or predicts outcomes by comparing new data points with similar examples.
    - **Decision Trees:** Uses a tree-structured model to make decisions based on splitting criteria.
    
    Each section allows you to interact with the models by uploading
    your own dataset or using a default one,
    tweaking parameters, and viewing performance metrics and visualizations. 
    Scroll through the tabs to see how these supervised learning techniques
    can be applied to real-world data.
    """)
