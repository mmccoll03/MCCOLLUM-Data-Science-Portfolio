import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage, dendrogram
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score


def show():
    st.title("Hierarchical Clustering")

    # Introduction
    st.markdown(
        """
        **Hierarchical Clustering** builds a tree of clusters (dendrogram) in a bottom-up manner (agglomerative).
        Each data point starts as its own cluster, and similar clusters merge iteratively.
        You can "cut" the dendrogram at various levels to see different numbers of clusters.
        """
    )

    # 1. Load Mall Customers Dataset
    st.subheader("1. Load Mall Customers Dataset")
    st.write(
        "We use the **Mall Customers** dataset (200 customers) to demonstrate hierarchical clustering. "
        "It contains: CustomerID, Gender, Age, Annual Income (k$), and Spending Score (1-100)."
        "While the other chapters allow you to upload a dataset, I thought" \
        " that might be a bit harder for this kind of algorithm. Still, " \
        "the mall customers dataset is well suited for this kind of clustering," \
        " and it isn't all that different from k-means clustering."
    )
    df = pd.read_csv("datasets/Mall_Customers.csv")
    st.dataframe(df.head())

    # 2. Exploratory Data Analysis
    st.subheader("2. Exploratory Data Analysis")
    st.write("Plot distributions of Age, Annual Income, and Spending Score.")
    fig_dist, axes = plt.subplots(1, 3, figsize=(15, 6))
    for i, col in enumerate(['Age', 'Annual Income (k$)', 'Spending Score (1-100)']):
        sns.histplot(df[col], bins=15, kde=True, ax=axes[i])
        axes[i].set_title(f"{col} Distribution")
    plt.tight_layout()
    st.pyplot(fig_dist)

    # 3. Label Encoding
    st.subheader("3. Label Encoding")
    st.write(
    """
    Convert the categorical 'Gender' column into numeric values using label encoding.
    Many machine learning algorithms, including hierarchical clustering, rely on numerical distance calculations between data points. 
    Since non-numeric values like strings cannot be directly compared using these distance metrics, we must first encode them as numbers. 
    This ensures that all features in the dataset can be processed uniformly and contribute meaningfully to the clustering process.
    """
    )
    le = preprocessing.LabelEncoder()
    df['Gender_enc'] = le.fit_transform(df['Gender'])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    st.write("Gender mapping:", mapping)

    # 4. Data Preparation
    st.subheader("4. Data Preparation")
    st.write(
        """
        Drop the 'CustomerID' column and assemble the feature matrix for clustering.
        The 'CustomerID' field is a unique identifier—it doesn't carry meaningful information about customer behavior or similarity. 
        Including it in clustering would introduce noise and distort distance calculations between data points.
        Instead, we focus on relevant numeric features like age, income, and spending score, which reflect measurable traits 
        we want the clustering algorithm to group by.
        """
    )
    features = ['Age', 'Annual Income (k$)', 'Spending Score (1-100)', 'Gender_enc']
    X = df[features]
    st.write("Feature matrix shape:", X.shape)
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # 5. Dendrogram with cluster slider
    st.subheader("5. Interactive Dendrogram")
    st.write(
        """
        Use the slider to choose how many clusters to highlight on the dendrogram.
        The dendrogram shows how customers are grouped together step by step based on their similarity in age, income, spending score, and gender. 
        Each merge in the tree reflects a decision to group two similar clusters.
        By adjusting the slider, you're selecting a level of the hierarchy at which to "cut" the tree—this defines how many clusters will be formed. 
        A higher cut (fewer clusters) reveals broad groupings, while a lower cut (more clusters) uncovers finer distinctions between customer segments.
        """
    )
    Z = linkage(X_std, method='ward')
    max_clusters = min(10, X_std.shape[0])
    dendro_k = st.slider(
        "Number of clusters for dendrogram cut:",
        min_value=2,
        max_value=max_clusters,
        value=5
    )
    # threshold to cut for k clusters
    # Compute threshold just above the distance that merges into k clusters
    threshold = Z[-(dendro_k - 1), 2]
    st.write(f"Cut dendrogram at distance = {threshold:.2f} to form {dendro_k} clusters.")
    fig_den, ax_den = plt.subplots(figsize=(16, 8))
    dendrogram(
        Z,
        leaf_rotation=90,
        leaf_font_size=8,
        ax=ax_den,
        color_threshold=threshold
    )
    ax_den.axhline(y=threshold, color='red', linestyle='--', label='Cut threshold')
    ax_den.set_xlabel('Sample Index')
    ax_den.set_ylabel('Distance')
    ax_den.legend()
    st.pyplot(fig_den)

    # 6. Agglomerative Clustering
    st.subheader("6. Form Flat Clusters & Counts")
    st.write(
        """
        Based on your selection from the dendrogram slider, this step performs agglomerative clustering to group customers into the specified number of clusters.
        Agglomerative clustering works by iteratively merging the closest pairs of data points or clusters, using the Ward linkage method—which minimizes variance within each cluster.
        Once clustering is complete, each customer is assigned a cluster label. The table below shows how many customers belong to each group, giving you a sense of how the segments are distributed.
        """
    )
    hc = AgglomerativeClustering(n_clusters=dendro_k, linkage='ward')
    labels = hc.fit_predict(X_std)
    df['cluster'] = labels
    counts = pd.Series(labels).value_counts().sort_index()
    st.write("Cluster counts:")
    st.dataframe(
        counts.rename_axis('Cluster').reset_index(name='Count')
    )

    # 7. 3D Cluster Visualization
    st.subheader("7. 3D Cluster Visualization")
    st.write("Age (x), Spending Score (y), Income (z); colored by cluster.")
    trace = go.Scatter3d(
        x=df['Age'],
        y=df['Spending Score (1-100)'],
        z=df['Annual Income (k$)'],
        mode='markers',
        marker=dict(
            size=6,
            color=labels,
            colorscale='Viridis',
            showscale=True,
            opacity=0.8
        )
    )
    fig3d = go.Figure(data=[trace])
    fig3d.update_layout(
        title='3D Agglomerative Clustering',
        scene=dict(
            xaxis_title='Age',
            yaxis_title='Spending Score (1-100)',
            zaxis_title='Annual Income (k$)'
        )
    )
    st.plotly_chart(fig3d)

    # 8. 2D Cluster Projection
    st.subheader("8. 2D Cluster Projection")
    st.write("Income vs. Spending Score colored by cluster.")
    fig2d, ax2d = plt.subplots(figsize=(8, 6))
    palette = sns.color_palette('tab10', dendro_k)
    for cluster in range(dendro_k):
        subset = df[df['cluster'] == cluster]
        ax2d.scatter(
            subset['Annual Income (k$)'],
            subset['Spending Score (1-100)'],
            s=50,
            c=[palette[cluster]],
            label=f'Cluster {cluster}',
            alpha=0.7,
            edgecolor='k'
        )
    ax2d.set_xlabel('Annual Income (k$)')
    ax2d.set_ylabel('Spending Score (1-100)')
    ax2d.set_title('Clusters of Mall Customers')
    ax2d.legend()
    st.pyplot(fig2d)

    # 9. Silhouette Analysis & Suggested k
    st.subheader("9. Silhouette Analysis & Suggested *k*")
    st.write(
        """
        To help choose the optimal number of clusters, we compute the average silhouette score for *k* ranging from 2 to 10.
        A higher silhouette score indicates tighter, better-separated clusters.
        """
    )

    # Compute silhouette scores for k=2..10
    k_values = list(range(2, 11))
    sil_scores = []
    for k in k_values:
        labels_k = AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_std)
        sil_scores.append(silhouette_score(X_std, labels_k))

    # Plot silhouette curve
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(k_values, sil_scores, marker="o")
    ax.set_xticks(k_values)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Average Silhouette Score")
    ax.set_title("Silhouette Analysis for Agglomerative Clustering")
    ax.grid(alpha=0.3)
    st.pyplot(fig)

    # Report best k
    best_k = k_values[int(np.argmax(sil_scores))]
    best_score = max(sil_scores)
    st.write(f"**Optimal number of clusters by silhouette:** {best_k}  (score = {best_score:.3f})")

    pass
