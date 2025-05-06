import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def show():
    st.title("Principal Component Analysis (PCA)")

    # Overview
    st.markdown(
        """
        **Principal Component Analysis (PCA)** transforms high-dimensional, correlated data 
        into a smaller set of uncorrelated components that capture the most variance.

        - **PC1** captures the largest share of variance.
        - **PC2** captures the next largest share, orthogonal to PC1.
        - And so on…
        
        PCA is invaluable for visualization, noise reduction, and speeding up downstream models 
        when your feature space is large.
        """
    )

    # 1. Load & Preprocess Data
    st.subheader("1. Load & Preprocess Data")
    source = st.radio(
        "Choose a data source:",
        ("Example: Breast Cancer", "Upload CSV")
    )

    if source == "Example: Breast Cancer":
        st.markdown(
            """
            **Breast Cancer Wisconsin Dataset**  
            569 samples × 30 numeric features (e.g. radius, texture, perimeter, area,…).  
            Labels are **benign** vs. **malignant**, but PCA ignores them during fitting.

            This dataset is perfect for PCA because:
            1. It’s moderately high-dimensional (30 features).  
            2. Many features are correlated, so PCs can capture shared structure.  
            3. It’s easy to visualize class separation in 2D/3D PC space.
            """
        )
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        y = pd.Series(data.target, name="target")
        labels = pd.Series(data.target).map({0: data.target_names[0], 1: data.target_names[1]})
        has_labels = True
    else:
        uploaded = st.file_uploader("Upload your CSV for PCA", type=["csv"])
        if not uploaded:
            st.info("Awaiting CSV file upload...")
            return
        df = pd.read_csv(uploaded).drop_duplicates().dropna()
        y = None
        has_labels = False

    st.write("Dataset preview:")
    st.dataframe(df.head())

    # Extract numeric columns
    numeric_df = df.select_dtypes(include=np.number)
    if numeric_df.shape[1] < 2:
        st.error("Need at least two numeric features for PCA.")
        return

    # 2. Feature Selection
    st.subheader("2. Feature Selection")
    variances = numeric_df.var().sort_values(ascending=False)
    feats = variances.index.tolist()
    default_k = min(5, len(feats))
    k_feats = st.slider("Top-variance features to include", 2, len(feats), default_k)
    selected = feats[:k_feats]
    st.write(f"Automatically selected (by variance): {selected}")

    with st.expander("Or select manually"):
        manual = st.multiselect("Pick features:", options=feats, default=selected)
    if manual:
        selected = manual
    st.write("Final features:", selected)

    X = numeric_df[selected]

    # 3. Standardize
    st.subheader("3. Standardize Features")
    st.write(
    """
    Standardizing the features to have zero mean and unit variance is essential before applying PCA. 

    Without scaling, features with larger numeric ranges could dominate the principal components, skewing the results. This step ensures that all features contribute equally to the analysis, regardless of their original scale or units.
    """
    )

    scaler = StandardScaler()
    X_std = scaler.fit_transform(X)

    # 4. Number of Components
    st.subheader("4. Number of Principal Components")
    max_c = min(len(selected), 10)
    n_comp = st.slider("How many PCs to compute?", 1, max_c, min(2, max_c))
    st.write(f"Computing {n_comp} principal components…")

    # 5. Fit & Transform
    pca = PCA(n_components=n_comp)
    scores = pca.fit_transform(X_std)
    exp_var = pca.explained_variance_ratio_
    cum_var = np.cumsum(exp_var)

    # 6. Scree Plot
    st.subheader("5. Scree Plot")
    st.write(
        """
        The scree plot helps visualize how much variance each principal component captures from the data.  
        - The **bars** represent the percentage of total variance explained by each individual component.  
        - The **line** shows the **cumulative variance**, helping you see how many components are needed to retain most of the information.

        This plot is especially useful for deciding how many principal components to keep.  
        Look for an **“elbow”**—a point where adding more components yields diminishing returns—indicating a good trade-off between dimensionality and information retained.
        """
    )
    fig1, ax1 = plt.subplots()
    ax1.bar(range(1, n_comp+1), exp_var*100, alpha=0.7, label="Individual (%)")
    ax1.plot(range(1, n_comp+1), cum_var*100, marker='o', color='orange', linestyle='--', label="Cumulative (%)")
    ax1.set_xlabel("Principal Component")
    ax1.set_ylabel("Variance Explained (%)")
    ax1.legend()
    st.pyplot(fig1)

    # 7. 2D Projection & Biplot
    if n_comp >= 2:
        st.subheader("6. 2D Projection")
        st.write(
        """
        This scatter plot shows the data projected onto the first two principal components (PC1 and PC2).  
        
        These components capture the directions of greatest variance in the data, and plotting them provides a compact, lower-dimensional view of the original dataset.

        If the dataset contains labeled classes, you can visually assess whether those classes are separable in this reduced feature space—which can be a valuable diagnostic before applying supervised models.
        """
        )
        proj = pd.DataFrame(scores[:, :2], columns=["PC1","PC2"])
        if has_labels:
            proj["label"] = labels.values
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=proj, x="PC1", y="PC2", hue="label", ax=ax2)
            ax2.set_title("PC1 vs PC2 by Label")
            st.pyplot(fig2)

            st.subheader("7. Biplot")
            st.write(
            """
            The biplot overlays arrows representing the contribution of each original feature  
            to the first two principal components.

            The **direction** of each arrow shows how strongly that feature influences PC1 and PC2,  
            and the **length** of the arrow indicates the strength of its contribution.  
            
            This helps interpret what each principal component represents in terms of the original features.
            """
            )
            loadings = pca.components_.T
            fig3, ax3 = plt.subplots(figsize=(8,6))
            sns.scatterplot(data=proj, x="PC1", y="PC2", hue="label", ax=ax3, alpha=0.7)
            scale = np.max(np.abs(proj[["PC1","PC2"]].values)) * 0.4
            for i, feat in enumerate(selected):
                ax3.arrow(0,0, loadings[i,0]*scale, loadings[i,1]*scale,
                          color='red', width=0.005, head_width=0.1)
                ax3.text(loadings[i,0]*scale*1.1, loadings[i,1]*scale*1.1,
                         feat, color='red', ha='center')
            ax3.set_title("Biplot")
            st.pyplot(fig3)
        else:
            fig2, ax2 = plt.subplots()
            sns.scatterplot(data=proj, x="PC1", y="PC2", ax=ax2)
            ax2.set_title("PC1 vs PC2")
            st.pyplot(fig2)

    # 8. Classification Comparison
    if has_labels:
        st.subheader("8. Classification Comparison")
        st.write(
            """
            Compare logistic regression accuracy on:
            - Original standardized features.
            - PCA-reduced data using your selected number of components.
            """
        )
        # split original and PCA once
        X_tr_o, X_te_o, y_tr, y_te = train_test_split(X_std, y, test_size=0.2, random_state=42)
        X_tr_p, X_te_p, _, _   = train_test_split(scores, y, test_size=0.2, random_state=42)

        clf_o   = LogisticRegression(max_iter=10000)
        clf_pca = LogisticRegression(max_iter=10000)
        clf_o.fit(X_tr_o, y_tr)
        clf_pca.fit(X_tr_p, y_tr)

        acc_o = accuracy_score(y_te, clf_o.predict(X_te_o))
        acc_p = accuracy_score(y_te, clf_pca.predict(X_te_p))

        st.write(f"- Original data accuracy: **{acc_o*100:.2f}%**")
        st.write(f"- PCA ({n_comp} components) accuracy: **{acc_p*100:.2f}%**")

    # 9. Show Loadings
    st.subheader("9. Feature Loadings")
    st.write("Each column shows how strongly a feature contributes to that PC.")
    load_df = pd.DataFrame(
        pca.components_.T,
        index=selected,
        columns=[f"PC{i+1}" for i in range(n_comp)]
    )
    st.dataframe(load_df)

    # 10. Download Scores
    st.subheader("10. Download PCA Scores")
    csv = pd.DataFrame(scores, columns=[f"PC{i+1}" for i in range(n_comp)]).to_csv(index=False)
    st.download_button("Download as CSV", csv, file_name="pca_scores.csv")
