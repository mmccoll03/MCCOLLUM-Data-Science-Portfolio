import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
import os


def show():
    st.title("K-Means Clustering")

    # Algorithm Explanation
    st.markdown(
        """
        **K-Means Clustering** is an unsupervised machine learning algorithm used to group data into *K* distinct clusters based on feature similarity.

        The goal is to partition the data so that points within the same cluster are as close as possible to each other, while clusters themselves are well separated.

        The algorithm works iteratively:
        1. Choose the number of clusters (*K*) and initialize *K* centroids (either randomly or with a smarter method like k-means++).
        2. Assign each data point to the nearest centroid using a distance metric (usually Euclidean distance).
        3. Update each centroid to be the mean of all points assigned to that cluster.
        4. Repeat steps 2 and 3 until the centroids stop changing or a maximum number of iterations is reached.

        K-Means is especially useful for segmenting data when the number of groups is known or can be estimated using evaluation methods like the elbow plot or silhouette score.
        """
    )


    # 1. Load & Preprocess Data
    st.subheader("1. Load & Preprocess Data")
    data_source = st.radio(
        "Choose a data source:",
        ("Example: Mall Customers", "Upload your own CSV")
    )

    if data_source == "Example: Mall Customers":
        @st.cache_data
        def load_example_data():
            # __file__ is something like ".../MLUnsupervisedApp/app_pages/kmeans_clustering.py"
            base = os.path.dirname(__file__)               # .../MLUnsupervisedApp/app_pages
            repo_root = os.path.abspath(os.path.join(base, os.pardir))   # .../MLUnsupervisedApp
            csv_path = os.path.join(repo_root, "datasets", "Mall_Customers.csv")
            return pd.read_csv(csv_path)
        df = load_example_data()

        # Dataset Description
        st.subheader("About the Mall Customers Dataset")
        st.write(
            """
            The **Mall Customers** dataset contains data on 200 customers of a shopping mall:
            
            - **CustomerID**: Unique identifier for each customer.
            - **Gender**: Male or Female.
            - **Age**: Customer's age in years.
            - **Annual Income (k$)**: Annual income of the customer in thousands of dollars.
            - **Spending Score (1-100)**: A score assigned by the mall based on customer behavior and spending patterns.
            
            You can use this widget to explore how different features group customers into segments. For example, select **Annual Income (k$)** and **Spending Score (1-100)** as your features, and set *K* between 3 and 6 to identify meaningful customer segments.
            """
        )

        # Basic preprocessing
        df = df.drop_duplicates().dropna()

        st.write("Dataset Preview after preprocessing:")
        st.dataframe(df.head())

        # 1a. Exploratory Data Analysis (EDA)
        st.subheader("1. Exploratory Data Analysis (EDA)")
        numeric_df = df.select_dtypes(include=np.number)
        numeric_cols = numeric_df.columns.tolist()

        if numeric_cols:
            # Histograms with KDE
            st.markdown("**Feature Distributions:** Histograms and KDE for numeric features")
            fig_hist, axes = plt.subplots(1, len(numeric_cols), figsize=(4 * len(numeric_cols), 4))
            for idx, col in enumerate(numeric_cols):
                sns.histplot(numeric_df[col], kde=True, ax=axes[idx])
                axes[idx].set_title(col)
            fig_hist.tight_layout()
            st.pyplot(fig_hist)

            # Gender countplot
            if 'Gender' in df.columns:
                st.markdown("**Gender Distribution:**")
                fig_count, ax_count = plt.subplots(figsize=(6, 2))
                sns.countplot(x='Gender', data=df, ax=ax_count)
                ax_count.set_title('Gender Distribution')
                st.pyplot(fig_count)

            # Boxplots by Gender
            if 'Gender' in df.columns:
                st.markdown("**Boxplots by Gender:**")
                fig_box, axes_box = plt.subplots(1, len(numeric_cols), figsize=(4 * len(numeric_cols), 4))
                for idx, col in enumerate(numeric_cols):
                    sns.boxplot(x=col, y='Gender', data=df, color='purple', ax=axes_box[idx])
                    axes_box[idx].set_title(col)
                fig_box.tight_layout()
                st.pyplot(fig_box)
    else:
        uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])
        if not uploaded_file:
            st.info("Awaiting CSV file upload.")
            return
        df = pd.read_csv(uploaded_file)
        df = df.drop_duplicates().dropna()

        st.write("Dataset Preview after preprocessing:")
        st.dataframe(df.head())

    # 2. Select Features & Hyperparameters
    numeric_df = df.select_dtypes(include=np.number)
    numeric_cols = numeric_df.columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least two numeric columns for clustering.")
        return

    st.subheader("2. Select Features & Hyperparameters")
    col_x = st.selectbox("X-axis feature", numeric_cols, index=0)
    col_y = st.selectbox("Y-axis feature", numeric_cols, index=1)
    k = st.slider("Number of clusters (K)", min_value=2, max_value=10, value=5)

    if st.button("Run K-Means Clustering"):
        X = numeric_df[[col_x, col_y]]

        # Standardize features
        scaler = StandardScaler()
        X_std = scaler.fit_transform(X)

        # Fit K-Means on standardized data
        kmeans = KMeans(n_clusters=k, random_state=42)
        labels = kmeans.fit_predict(X_std)
        df["Cluster"] = labels

        # Inverse transform cluster centers to original scale
        centers_std = kmeans.cluster_centers_
        centers = scaler.inverse_transform(centers_std)
        centers_df = pd.DataFrame(centers, columns=[col_x, col_y])

        # Plot clusters
        fig1, ax1 = plt.subplots()
        sns.scatterplot(
            x=col_x,
            y=col_y,
            hue=labels,
            palette="tab10",
            data=df,
            ax=ax1
        )
        ax1.set_title(f"K-Means Clustering (K={k})")
        st.pyplot(fig1)

        st.subheader("Cluster Centers (Original Scale)")
        st.dataframe(centers_df)

        # Educational explanations
        st.markdown(
            "### How the Elbow Method Works\n"
            "The elbow method plots the **Within-Cluster Sum of Squares (WCSS)** against *k*. "
            "WCSS measures cluster compactness: a lower value is better. "
            "As *k* increases, WCSS decreases—but look for the 'elbow' where improvement slows, indicating optimal *k*. This should look like a kink in the graph."
        )
        st.markdown(
            "### Silhouette Score\n"
            "Silhouette score measures how well each point lies within its cluster vs others, ranging from -1 to 1:\n"
            "- **1.0**: Perfectly clustered.\n"
            "- **0.0**: On the boundary between clusters.\n"
            "- **-1.0**: Likely misassigned.\n"
            "A higher average silhouette suggests more distinct clusters."
        )

        # 3. Evaluation Metrics: Elbow & Silhouette
        st.subheader("3. Evaluation Metrics: Elbow & Silhouette")
        ks = range(2, 11)
        wcss = []
        sil_scores = []
        for ki in ks:
            km = KMeans(n_clusters=ki, random_state=42)
            km.fit(X_std)
            wcss.append(km.inertia_)
            sil_scores.append(silhouette_score(X_std, km.labels_))

        fig2, (ax2, ax3) = plt.subplots(1, 2, figsize=(12, 4))
        ax2.plot(ks, wcss, marker='o')
        ax2.set_xlabel('Number of clusters (k)')
        ax2.set_ylabel('WCSS')
        ax2.set_title('Elbow Graph')
        ax2.grid(True)

        ax3.plot(ks, sil_scores, marker='o')
        ax3.set_xlabel('Number of clusters (k)')
        ax3.set_ylabel('Silhouette Score')
        ax3.set_title('Silhouette Scores')
        ax3.grid(True)

        fig2.tight_layout()
        st.pyplot(fig2)
