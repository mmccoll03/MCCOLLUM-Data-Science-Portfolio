import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

def show_model_page(title):
    st.header("Model Visualization")

    # Define parameter sliders in two columns
    col1, col2 = st.columns(2)
    with col1:
        sample_size = st.slider("Sample Size", min_value=100, max_value=10000, value=1000, step=100)
        error_noise = st.slider("Error (ε)", min_value=0.0, max_value=10.0, value=2.0, step=0.1)
    with col2:
        intercept_val = st.slider("Intercept", min_value=-10.0, max_value=10.0, value=0.0, step=0.1)
        slope_val = st.slider("Slope", min_value=-5.0, max_value=5.0, value=1.0, step=0.1)

    # Create synthetic data for the regression model
    x = np.linspace(-5, 10, sample_size).reshape(-1, 1)
    noise = error_noise * np.random.randn(sample_size)
    y = intercept_val + slope_val * x.flatten() + noise

    # Fit the OLS model using statsmodels
    X_with_const = sm.add_constant(x)
    model = sm.OLS(y, X_with_const).fit()

    # Display the Regression Plot
    st.subheader("Regression Plot")
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(x, y, alpha=0.5, color='#5b1ce3', label='Data Points')
    ax.plot(x, model.predict(X_with_const), color='#ed0100', linewidth=2, label='Fitted Line')
    
    
    ax.set_xlabel("X (Predictor)")
    ax.set_ylabel("Y (Response)")
    ax.set_title("Linear Regression Fit")
    ax.legend()
    st.pyplot(fig)

    # Expander to display the model summary
    with st.expander("Click to View the Model Summary"):
        st.markdown("""
        <div style='text-align: center;'>
            <h2 style='font-size: 24px;'>Model Summary</h2>
        </div>
        """, unsafe_allow_html=True)
        
        # Display the full model summary as a preformatted text block
        summary_text = model.summary().as_text()
        st.code(summary_text, language='python')

def show_data_upload_and_model():
    st.header("Dataset Upload and Model Evaluation")
    st.write("""
    In this section, you can either upload your own dataset (CSV file) or use the default Wine dataset.
    After the data is loaded, select a dependent variable (target) and one or more independent variables (features).
    Adjust the sliders as needed, then train a model to see evaluation metrics such as R² and Adjusted R².
    """)
    
    # Option to choose data source
    data_source = st.radio("Select Data Source", ("Upload Your Own Dataset", "Use Wine Dataset"))

    df = None
    if data_source == "Upload Your Own Dataset":
        uploaded_file = st.file_uploader("Upload CSV", type="csv")
        if uploaded_file is not None:
            df = pd.read_csv(uploaded_file)
    else:
        # Replace the file path with the actual path if needed.
        try:
            df = pd.read_csv("datasets/wine.csv")
        except Exception as e:
            st.error("Error loading wine dataset. Please check the file path.")
    
    if df is not None:
        st.write("### Data Preview")
        st.dataframe(df.head())
        
        # Let user select the dependent variable
        target = st.selectbox("Select Dependent Variable (Target)", options=df.columns)
        
        # Let user select one or more independent variables
        features = st.multiselect(
            "Select Independent Variable(s) (Features) (Note: Choose variables that are numerical or can work as discrete variables)",
            options=[col for col in df.columns if col != target]
        )
        
        # TODO: Add any dataset-specific preprocessing, filtering, or slider options here.
        
        # Proceed only if at least one feature is selected
        if features:
            # Drop rows with missing values in the selected columns
            df_clean = df.dropna(subset=[target] + features)
            if df_clean.empty:
                st.error("No data available after removing missing values from the selected columns.")
                return
            
            # Optionally, choose a test size for splitting the data
            test_size = st.slider("Test Size (%)", min_value=10, max_value=50, value=30, step=5) / 100.0
            
            # Split the dataset into training and test sets
            X = df_clean[features]
            y = df_clean[target]
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
            
            # Train a simple linear regression model as an example
            from sklearn.linear_model import LinearRegression
            model = LinearRegression()
            model.fit(X_train, y_train)
            r_squared = model.score(X_test, y_test)
            
            # Calculate Adjusted R²
            n = X_test.shape[0]
            p = X_test.shape[1]
            adjusted_r_squared = 1 - (1 - r_squared) * (n - 1) / (n - p - 1) if n > p + 1 else r_squared

            st.subheader("Model Performance")
            st.write(f"**R²:** {r_squared:.3f}")
            st.write(f"**Adjusted R²:** {adjusted_r_squared:.3f}")
            
            # Plot predictions vs actual values
            y_pred = model.predict(X_test)
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.scatter(y_test, y_pred, alpha=0.6, color="#8a2be2", label="Predicted vs Actual")
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Model Predictions")
            ax.legend()
            st.pyplot(fig)
            
            # Generate and display the model equation in LaTeX format using the selected variables
            st.subheader("Model Equation")
            equation = f"y = {model.intercept_:.3f}"
            for coef, feat in zip(model.coef_, features):
                sign = " + " if coef >= 0 else " - "
                equation += f"{sign}{abs(coef):.3f}\\,{feat}"
            equation += " + \\epsilon"
            st.latex(equation)
            
            # TODO: Add additional evaluation metrics (e.g., Precision, if applicable) or visualizations as needed.
        else:
            st.write("No model created. Please select at least one feature (independent variable) for modeling.")




def show():
    # Title and header with custom styling
    st.markdown("""
    <h1 style='text-align: center; font-family: Garamond, serif; color: #f0f0f0;'>
        Linear Regression
    </h1>
    """, unsafe_allow_html=True)

    # Brief introduction to linear regression
    st.write("""
    Linear Regression is a foundational statistical tool that describes the relationship between a dependent variable and one or more independent predictors. By finding the best-fit line through your data, this method minimizes the prediction error and provides insights into how the predictors affect the outcome.
    """)

    # Section: Key Concepts
    st.write("## Key Concepts")
    st.write("""
    At its heart, Linear Regression models the relationship using a linear equation. The basic idea is to explain the variation in the dependent variable using the changes in the predictors.
    """)

    # Display the core equation with LaTeX formatting
    st.write("### Simple Linear Regression Equation")
    st.latex(r"y = \beta_0 + \beta_1x + \epsilon")
    st.write("""
    - **y**: Dependent variable (what you want to predict)
    - **x**: Independent variable (the predictor)
    - **β₀**: Intercept of the line (the baseline value when x = 0)
    - **β₁**: Slope (how much y changes per unit change in x)
    - **ε**: Error term (captures unexplained variability)
    """)

    # Section: Assumptions Underlying the Model
    st.write("## Fundamental Assumptions")
    st.write("""
    For reliable results, the model makes several important assumptions:
    1. **Linearity:** The relationship between predictors and the outcome is linear.
    2. **Independence:** The residuals (errors) are independent.
    3. **Constant Variance:** The variance of residuals remains steady across all levels of x (homoscedasticity).
    4. **Normality:** The residuals follow a normal distribution.
    """)

    # Section: Parameter Estimation and Model Evaluation
    st.write("## Model Fitting and Evaluation")
    st.write("""
    The most common method for estimating the parameters (β₀ and β₁) is the Ordinary Least Squares (OLS). OLS minimizes the sum of squared residuals:
    """)
    st.latex(r"\min_{\beta_0, \beta_1} \sum_{i=1}^n (y_i - (\beta_0 + \beta_1x_i))^2")
    st.write("""
    Once fitted, the model can be evaluated using metrics such as R-squared, Adjusted R-squared, and analyzing residual plots.
    """)

    # Section: Multiple Linear Regression
    st.write("## Multiple Linear Regression")
    st.write("""
    When your model includes more than one predictor, the regression equation expands to account for all the independent variables. This is known as Multiple Linear Regression.
    """)
    st.latex(r"y = \beta_0 + \beta_1x_1 + \beta_2x_2 + \cdots + \beta_p x_p + \epsilon")
    st.write("""
    In this formulation:
    - **y** represents the predicted response.
    - **β₀** is the intercept, indicating the baseline value of y when all predictors are zero.
    - **β₁, β₂, …, βₚ** are the coefficients for each predictor (**x₁, x₂, …, xₚ**), quantifying the change in the response for a one-unit change in the corresponding predictor (while keeping the others constant).
    - **ε** accounts for the error or noise not explained by the model.

    This approach allows you to evaluate the combined impact of several independent variables on the outcome.
    """)

    # Section: Limitations and Extensions
    st.write("## Practical Considerations")
    st.write("""
    While Linear Regression is powerful, it may falter when:
    - The relationship is not strictly linear.
    - Outliers skew the results.
    - Predictors are highly correlated (multicollinearity).
    
    In such cases, alternative models or enhancements like regularization or transformations might be more appropriate.
    """)

    show_model_page(
        "Linear Regression",
    )

    show_data_upload_and_model()