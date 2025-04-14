import streamlit as st
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import graphviz

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, roc_auc_score

def show():
    st.markdown("""
    <h1 style='text-align: center; font-family: Garamond, serif; color: #f0f0f0;'>
        Decision Tree Classification
    </h1>
    """, unsafe_allow_html=True)
    
    st.write("""
    **About Decision Trees:**

    Decision trees are a versatile supervised learning method used for both classification and regression. 
    They work by recursively splitting the dataset based on feature values to create a tree-like model of decisions.
    The splits are chosen to maximize the separation of classes (or to reduce variance in regression).
    In classification tasks, decision trees can provide an intuitive visualization of how decisions are made. 
    In this interactive example, you can adjust the maximum tree depth to see how it affects model performance on the Titanic dataset.
    You may also upload your own dataset. There are also measures like the gini index, which measure split
    diversity.
    """)

    # -------------------------------------
    # DATA LOADING AND PREPROCESSING
    # -------------------------------------
    data_source = st.radio("Select Data Source", options=["Upload Your Own Dataset", "Use Titanic Dataset (Default)"])
    df = None

    if data_source == "Upload Your Own Dataset":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
    else:
        # Load Titanic dataset from seaborn and process
        df = sns.load_dataset('titanic')
    
    if df is not None:
        st.subheader("Data Preview")
        st.dataframe(df.head())
        
        # If using the Titanic dataset, perform preprocessing.
        if data_source != "Upload Your Own Dataset":
            # Drop rows with missing age values
            df.dropna(subset=['age'], inplace=True)
            # One-hot encode categorical variable 'sex' (drop first to avoid dummy trap)
            df = pd.get_dummies(df, columns=['sex'], drop_first=True)
            # Define default features and target for Titanic
            default_features = ['pclass', 'age', 'sibsp', 'parch', 'fare', 'sex_male']
            default_target = 'survived'
        else:
            # For an uploaded dataset, let the user select target and features.
            default_features = list(df.columns)
            default_target = None

        # -------------------------------------
        # Variable Selection
        # -------------------------------------
        if data_source == "Upload Your Own Dataset":
            target = st.selectbox("Select Dependent Variable (Target)", options=df.columns)
            features = st.multiselect("Select Independent Variable(s) (Features)", 
                                      options=[col for col in df.columns if col != target])
            if not features:
                st.info("Please select at least one feature for modeling.")
                return
        else:
            target = default_target
            features = default_features
            # Remove the target from features if needed
            if target in features:
                features.remove(target)
        
        st.write(f"**Target:** {target}")
        st.write(f"**Features:** {features}")

        # -------------------------------------
        # Train-Test Split and Optional Scaling (if needed)
        # -------------------------------------
        X = df[features]
        y = df[target]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # -------------------------------------
        # Decision Tree Training with Interactivity
        # -------------------------------------
        st.markdown("### Model Training Parameters")
        max_depth = st.slider("Select Maximum Tree Depth", min_value=1, max_value=10, value=5, step=1)
        
        # Initialize and train the decision tree classifier
        dt_model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
        dt_model.fit(X_train, y_train)
        y_pred = dt_model.predict(X_test)
        accuracy_val = accuracy_score(y_test, y_pred)
        
        st.write(f"**Accuracy:** {accuracy_val:.2f}")
        
        # Compute additional metrics and get classification report
        cm = confusion_matrix(y_test, y_pred)
        clf_report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(clf_report).transpose()
        
        # If target is binary, compute ROC and AUC
        if len(np.unique(y)) == 2:
            y_probs = dt_model.predict_proba(X_test)[:, 1]
            fpr, tpr, thresholds = roc_curve(y_test, y_probs)
            auc_val = roc_auc_score(y_test, y_probs)
        else:
            fpr, tpr, auc_val = None, None, None

        # -------------------------------------
        # Decision Tree Visualization
        # -------------------------------------
        st.subheader("Decision Tree Visualization")
        try:
            dot_data = export_graphviz(dt_model, out_file=None, feature_names=X_train.columns,
                                       class_names=[str(cls) for cls in np.unique(y)],
                                       filled=True, rounded=True, special_characters=True)
            st.graphviz_chart(dot_data)
        except Exception as e:
            st.error("Graphviz visualization could not be generated.")
        
        # ----------------------------------
        # ROC Curve (for binary classification)
        # -------------------------------------
        if fpr is not None and tpr is not None:
            st.subheader("ROC Curve")
            fig_roc, ax_roc = plt.subplots(figsize=(8, 6))
            ax_roc.plot(fpr, tpr, color="#8a2be2", lw=2, label=f"ROC (AUC = {auc_val:.2f})")
            ax_roc.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random Classifier")
            ax_roc.set_xlabel("False Positive Rate")
            ax_roc.set_ylabel("True Positive Rate")
            ax_roc.set_title("Receiver Operating Characteristic (ROC) Curve")
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)

        # -------------------------------------
        # Display Classification Report
        st.subheader("Classification Report")
        st.dataframe(report_df)

        # -------------------------------------
        # Additional Data Information
        # -------------------------------------
        with st.expander("Click to View Data Information"):
            st.write("### Overview of the Dataset")
            st.write("""
                This dataset is either your uploaded file or the default Titanic dataset. The default Titanic dataset includes passenger information, 
                such as passenger class (pclass), age, number of siblings/spouses aboard (sibsp), number of parents/children aboard (parch),
                fare, and gender (encoded as 'sex_male') for classification of survival.
            """)
            st.write("#### First 5 Rows")
            st.dataframe(df.head())
            st.write("#### Statistical Summary")
            st.dataframe(df.describe())
