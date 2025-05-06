import streamlit as st

def show():
    # Main title
    st.title("Unsupervised Machine Learning Portfolio")

    # Overview
    st.write(
        """
        Welcome to your interactive portfolio for **Unsupervised Machine Learning**. 

        Unlike supervised learning, unsupervised methods uncover patterns in data without relying on predefined labels. These techniques are often used for tasks like dimensionality reduction, data exploration, and customer segmentation.

        This app showcases three foundational unsupervised learning methods:

        - **Principal Component Analysis (PCA):** A method for reducing the number of features while retaining as much variability in the data as possible. PCA helps simplify complex datasets and is useful for visualization and preprocessing.

        - **K-Means Clustering:** A popular algorithm that groups data into *k* clusters based on similarity. It's widely used in market segmentation, image compression, and more.

        - **Hierarchical Clustering:** This approach builds a hierarchy of nested clusters and represents them in a dendrogram. It's especially helpful when you want to visualize how data groups at different levels of granularity.
        """
    )

    # How to use
    st.write(
        """
        ### How to Use This App

        Use the navigation bar on the left to switch between the three modules. In each section, you'll be able to:

        - Upload your own dataset or work with a built-in example
        - Adjust key parameters like the number of clusters or components
        - Explore visual outputs such as cluster scatterplots, dendrograms, or PCA projections
        - View different evaluation metrics like elbow graphs, scree plots, and silohuette scores.

        This app is designed to provide a hands-on understanding of how these algorithms behave with real data. Whether you're reviewing concepts or exploring new datasets, the goal is to make these unsupervised methods more tangible and intuitive.
        """
    )
