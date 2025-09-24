import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_auc_score, roc_curve, f1_score, precision_score, recall_score
)
from sklearn.preprocessing import StandardScaler

def show():
    st.title("K-Nearest Neighbors (KNN) Classification on Loans Data")
    st.markdown("""
    K-nearest neighbors (KNN) is a simple algorithm that makes predictions based on similarity.

    Suppose you have a dataset where each entry includes some features (like height and weight,
    or temperature and humidity) and a known outcome or label.

    When a new data point comes in, KNN looks for the k closest points in the dataset—
    measured by how similar their features are—and checks the labels of those neighbors.

    If you're doing classification, it assigns the new point the most common label among its neighbors.
    If you're doing regression, it averages their values.

    KNN doesn’t build a model in advance—it just stores the data and makes decisions by comparing
    new inputs to known ones, relying on the idea that similar things tend to have similar outcomes.
                

    This interactive application demonstrates the performance of a K-Nearest Neighbors (KNN) classifier.
    You can either upload your own dataset (CSV file) or use the default **loans.csv** dataset (from the R tidyverse).
    
    The app allows you to:
    - **Select the dependent variable (target) and independent variables (features).**
    - **Toggle between unscaled and scaled data** to see the impact of feature scaling on performance.
    - **Adjust the number of neighbors (k)** used in the KNN classifier.
    - **View performance metrics** including accuracy, F1 score, precision, sensitivity (recall), specificity, and—for binary targets—ROC and AUC.
    """)

    # -------------- Data Source Selection --------------
    data_source = st.radio("Select Data Source", options=["Upload Your Own Dataset", "Use Loans Dataset"])
    df = None

    if data_source == "Upload Your Own Dataset":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
    else:
        # Load loans.csv directly from GitHub raw URL
        url = "https://raw.githubusercontent.com/mmccoll03/MCCOLLUM-Data-Science-Portfolio/main/MLStreamlitApp/datasets/loans.csv"
        try:
            df = pd.read_csv(url)
            st.success("Loaded loans.csv from GitHub successfully!")
        except Exception as e:
            st.error(f"Failed to load loans.csv from GitHub. Error: {e}")

    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head())

        # -------------- Variable Selection --------------
        target = st.selectbox("Select Dependent Variable (Target)(I would recommend not.fully.paid)", options=df.columns)
        features = st.multiselect("Select Independent Variable(s) (Features)", 
                                  options=[col for col in df.columns if col != target],
                                  help="Choose variables that are numeric or discrete.")
        if not features:
            st.info("Please select at least one feature for modeling.")
            return

        # Warn user if target is not binary (binary metrics may only be computed for binary tasks)
        if len(np.unique(df[target])) != 2:
            st.warning("The target variable is not binary. Metrics such as ROC/AUC, sensitivity, and specificity may not be computed.")

        # -------------- Data Preprocessing --------------
        data_type = st.radio("Data Type", options=["Unscaled", "Scaled"])
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        if data_type == "Scaled":
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train)
            X_test = scaler.transform(X_test)

        # -------------- Model Training --------------
        k = st.slider("Select Number of Neighbors (k, odd values recommended)", min_value=1, max_value=21, step=2, value=5)
        knn_model = KNeighborsClassifier(n_neighbors=k)
        knn_model.fit(X_train, y_train)
        y_pred = knn_model.predict(X_test)

        # -------------- Metrics Calculation --------------
        accuracy_val = accuracy_score(y_test, y_pred)
        # For binary classification use 'binary'; else, 'weighted'
        average_type = 'binary' if len(np.unique(y)) == 2 else 'weighted'
        f1_val = f1_score(y_test, y_pred, average=average_type)
        precision_val = precision_score(y_test, y_pred, average=average_type)
        recall_val = recall_score(y_test, y_pred, average=average_type)

        # Sensitivity is recall for the positive class; specificity = TN / (TN + FP)
        cm = confusion_matrix(y_test, y_pred)
        if cm.shape == (2,2):
            TN, FP, FN, TP = cm.ravel()
            specificity_val = TN / (TN + FP) if (TN + FP) > 0 else np.nan
        else:
            specificity_val = None

        # For binary targets, compute ROC and AUC
        if len(np.unique(y)) == 2:
            y_prob = knn_model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_prob)
            auc_val = roc_auc_score(y_test, y_prob)
        else:
            fpr, tpr, thresholds = None, None, None
            auc_val = None

        # -------------- Display Performance Metrics --------------
        st.subheader("Model Performance Metrics")
        st.write(f"**Accuracy:** {accuracy_val:.3f}")
        st.write(f"**F1 Score:** {f1_val:.3f}")
        st.write(f"**Precision:** {precision_val:.3f}")
        st.write(f"**Sensitivity (Recall):** {recall_val:.3f}")
        if specificity_val is not None:
            st.write(f"**Specificity:** {specificity_val:.3f}")
        if auc_val is not None:
            st.write(f"**ROC AUC:** {auc_val:.3f}")

        # ROC Curve Plot (for binary classification)
        if fpr is not None and tpr is not None:
            st.subheader("ROC Curve")
            fig_roc, ax_roc = plt.subplots(figsize=(6,4))
            ax_roc.plot(fpr, tpr, color="#8a2be2", label=f"ROC Curve (AUC = {auc_val:.3f})")
            ax_roc.plot([0, 1], [0, 1], "k--", label="Random Classifier")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("ROC Curve")
            ax_roc.legend()
            st.pyplot(fig_roc)

        # Display the confusion matrix and classification report side-by-side
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Confusion Matrix")
            fig_cm, ax_cm = plt.subplots(figsize=(6,4))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel("Predicted")
            ax_cm.set_ylabel("Actual")
            st.pyplot(fig_cm)
        with col2:
            st.subheader("Classification Report")
            st.write("Note that accuracy is just one number - I couldn't" \
            " figure out a nice way to show this table. It's better than the default way" \
            " it was showing in streamlit though.")
            # Compute the classification report as a dictionary
            report_dict = classification_report(y_test, y_pred, output_dict=True)
            # Convert the dictionary to a DataFrame and transpose it so that rows correspond to labels
            report_df = pd.DataFrame(report_dict).transpose()

            st.dataframe(report_df)

        # Additional Data Information
        with st.expander("Click to View Data Information"):
            st.write("#### Overview of the Loans Dataset")
            st.write("""
            The loans dataset typically contains information on loans such as amount, term, interest rates, borrower characteristics, and whether the loan was repaid or defaulted.
            You may add additional preprocessing steps here if needed.
            """)
            st.write("#### First 5 Rows of the Dataset")
            st.dataframe(df.head())
            st.write("#### Statistical Summary")
            st.dataframe(df.describe())
