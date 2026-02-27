import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score, mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns

# Models
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.svm import SVC, SVR

st.set_page_config(page_title="ML Model Selector", layout="wide")

st.title("🚀 Machine Learning Model Trainer")

uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # create tabs for analysis and modeling
    analysis_tab, model_tab = st.tabs(["🔍 Analysis", "🧠 Modeling"])

    with analysis_tab:
        st.subheader("📂 Dataset Preview")
        st.dataframe(df)

        # Descriptive analysis
        st.subheader("📊 Descriptive Analysis")
        with st.expander("Show summary statistics"):
            st.write(df.describe(include='all').transpose())
        if not df.select_dtypes(include=['object']).empty:
            with st.expander("Categorical value counts"):
                cat_cols = df.select_dtypes(include=['object']).columns.tolist()
                for col in cat_cols:
                    st.write(f"**{col}**")
                    st.write(df[col].value_counts())
        if df.select_dtypes(include=[np.number]).shape[1] > 1:
            with st.expander("Correlation matrix"):
                corr = df.select_dtypes(include=[np.number]).corr()
                fig, ax = plt.subplots()
                sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
                st.pyplot(fig)
            # Macro-level scatter pairplot (macro graph)
            with st.expander("Macro graph (pairplot)"):
                num_df = df.select_dtypes(include=[np.number])
                # limit to max 10 columns to keep it readable
                if num_df.shape[1] > 10:
                    num_df = num_df.iloc[:, :10]
                    st.write("Showing first 10 numeric features for pairplot")
                pair_fig = sns.pairplot(num_df)
                st.pyplot(pair_fig.fig)

    with model_tab:
        # Step 1: Choose Classification or Regression
        st.subheader("🎯 Step 1: Select Task Type")
        task_type = st.radio(
            "What type of problem are you solving?",
            ("Classification", "Regression"),
            horizontal=True
        )
        
        # Step 2: Model Selection based on task type (MOVED UP)
        st.subheader("🤖 Step 2: Select Model")
        
        if task_type == "Classification":
            model_name = st.selectbox(
                "Choose a Classification Model:",
                (
                    "Gaussian Naive Bayes",
                    "Logistic Regression",
                    "KNN",
                    "Decision Tree",
                    "Random Forest",
                    "SVM"
                )
            )
        else:
            model_name = st.selectbox(
                "Choose a Regression Model:",
                (
                    "Linear Regression",
                    "Ridge Regression",
                    "Lasso Regression",
                    "KNN",
                    "Decision Tree",
                    "Random Forest",
                    "SVM"
                )
            )

        # Step 3: Target Selection - Filter based on task type
        st.subheader("🎯 Step 3: Select Target Variable")
        
        if task_type == "Classification":
            # For classification, show only categorical columns
            categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
            if len(categorical_cols) == 0:
                st.warning("⚠️ No categorical columns found. Please select a dataset with categorical target variables for classification.")
                st.stop()
            target = st.selectbox("Select the categorical target variable:", categorical_cols)
        else:
            # For regression, show only numeric columns
            numeric_cols = df.select_dtypes(include=['float64', 'float32', 'int64', 'int32']).columns.tolist()
            if len(numeric_cols) == 0:
                st.warning("⚠️ No numeric columns found. Please select a dataset with numeric target variables for regression.")
                st.stop()
            target = st.selectbox("Select the numeric target variable:", numeric_cols)

        # Step 4: Feature Selection
        st.subheader("📌 Step 4: Select Features")
        feature_options = df.columns.drop(target)

        selected_features = st.multiselect(
            "Select feature columns:",
            feature_options,
            default=list(feature_options)
        )

        if len(selected_features) == 0:
            st.warning("Please select at least one feature.")
            st.stop()

        X = df[selected_features].copy()
        y = df[target].copy()

        # Encode categorical features in X
        categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            st.info(f"🔄 Encoding categorical features: {', '.join(categorical_cols)}")
            label_encoders = {}
            for col in categorical_cols:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col])
                label_encoders[col] = le
        
        # Encode target if it's categorical (string)
        le_target = None
        if y.dtype == 'object':
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)

        # Convert X to numeric
        X = X.astype(float)
        
        # For classification, ensure y is integer
        if task_type == "Classification":
            y = y.astype(int)

        # Train-Test Split
        st.subheader("📊 Step 5: Train-Test Split")
        test_size = st.slider(
            "Select Test Size (%)",
            min_value=10,
            max_value=50,
            value=20
        ) / 100

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

        # Step 6: Configure and Train Model
        st.subheader("🚀 Step 6: Train Model")

        if task_type == "Classification":
            if model_name == "Gaussian Naive Bayes":
                model = GaussianNB()

            elif model_name == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)

            elif model_name == "KNN":
                k = st.slider("Select K value", 1, 15, 5)
                model = KNeighborsClassifier(n_neighbors=k)

            elif model_name == "Decision Tree":
                depth = st.slider("Select Max Depth", 1, 20, 5)
                model = DecisionTreeClassifier(max_depth=depth)

            elif model_name == "Random Forest":
                n_trees = st.slider("Number of Trees", 10, 200, 100)
                model = RandomForestClassifier(n_estimators=n_trees, random_state=42)

            elif model_name == "SVM":
                model = SVC()

            # Train Model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Display Results
            accuracy = accuracy_score(y_test, y_pred)
            st.subheader("✅ Model Performance")
            st.success(f"Accuracy: {accuracy:.4f}")

            # Confusion Matrix
            st.subheader("📌 Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
            plt.xlabel("Predicted")
            plt.ylabel("Actual")
            st.pyplot(fig)

        else:  # Regression
            if model_name == "Linear Regression":
                model = LinearRegression()

            elif model_name == "Ridge Regression":
                alpha = st.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0)
                model = Ridge(alpha=alpha)

            elif model_name == "Lasso Regression":
                alpha = st.slider("Alpha (Regularization Strength)", 0.01, 10.0, 1.0)
                model = Lasso(alpha=alpha, max_iter=2000)

            elif model_name == "KNN":
                k = st.slider("Select K value", 1, 15, 5)
                model = KNeighborsRegressor(n_neighbors=k)

            elif model_name == "Decision Tree":
                depth = st.slider("Select Max Depth", 1, 20, 5)
                model = DecisionTreeRegressor(max_depth=depth)

            elif model_name == "Random Forest":
                n_trees = st.slider("Number of Trees", 10, 200, 100)
                model = RandomForestRegressor(n_estimators=n_trees, random_state=42)

            elif model_name == "SVM":
                model = SVR()

            # Train Model
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)

            # Display Results
            r2 = r2_score(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            
            st.subheader("✅ Model Performance")
            col1, col2, col3 = st.columns(3)
            col1.metric("R² Score", f"{r2:.4f}")
            col2.metric("MAE", f"{mae:.4f}")
            col3.metric("RMSE", f"{rmse:.4f}")

            # Predicted vs Actual
            st.subheader("📊 Predicted vs Actual Values")
            fig, ax = plt.subplots()
            ax.scatter(y_test, y_pred, alpha=0.5)
            ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
            ax.set_xlabel("Actual Values")
            ax.set_ylabel("Predicted Values")
            ax.set_title("Regression: Predicted vs Actual")
            st.pyplot(fig)


else:
    st.info("Please upload a CSV file to begin.")