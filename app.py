# app.py — Heart Disease EDA & ML Streamlit App
# Author: Arka Patra

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ------------------------
# Streamlit Configuration
# ------------------------
st.set_page_config(
    page_title="Heart Disease Prediction Dashboard",
    page_icon="❤️",
    layout="wide"
)

st.title("Insurence Prediction & EDA App")
st.markdown("### Upload a dataset to explore EDA , feature engineering, and model training interactively.")

# ------------------------
# File Upload Section
# ------------------------
uploaded_file = st.file_uploader("📤 Upload your CSV dataset", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.success("✅ Dataset uploaded successfully!")

    # ------------------------
    # Dataset Overview
    # ------------------------
    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### 📊 Basic Info")
    st.write(f"**Shape:** {df.shape}")
    st.write(f"**Columns:** {list(df.columns)}")
    st.write(df.describe())

    # ------------------------
    # Exploratory Data Analysis
    # ------------------------
    st.subheader("🔍 Exploratory Data Analysis (EDA)")

    # Correlation Heatmap (fixed version)
    st.write("#### Correlation Heatmap")

    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=["int64", "float64"])

    if not numeric_df.empty:
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)
    else:
        st.warning("⚠️ No numeric columns found for correlation heatmap.")


    # Missing Values
    st.write("#### Missing Values Summary")
    st.write(df.isnull().sum())

    # ------------------------
    # Feature Engineering
    # ------------------------
    st.subheader("⚙️ Feature Engineering")

    # Encoding categorical variables
    cat_cols = df.select_dtypes(include=["object"]).columns
    if len(cat_cols) > 0:
        st.write("Encoding categorical columns:", list(cat_cols))
        df_encoded = pd.get_dummies(df, drop_first=True)
    else:
        df_encoded = df.copy()
        st.write("No categorical columns found — skipping encoding.")

    st.write("Shape after encoding:", df_encoded.shape)

    # Feature scaling
    st.subheader("📏 Feature Scaling")
    numeric_cols = df_encoded.select_dtypes(include=["int64", "float64"]).columns
    if len(numeric_cols) > 0:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(df_encoded[numeric_cols])
        df_scaled = pd.DataFrame(scaled_data, columns=numeric_cols)
        st.write(df_scaled.head())
    else:
        st.warning("No numeric columns found for scaling.")

    # ------------------------
    # PCA Visualization
    # ------------------------
    st.subheader("🧠 PCA (Principal Component Analysis)")
    if len(numeric_cols) > 1:
        pca = PCA(n_components=2)
        pca_data = pca.fit_transform(df_encoded[numeric_cols])
        df_pca = pd.DataFrame(pca_data, columns=["PC1", "PC2"])

        fig, ax = plt.subplots()
        ax.scatter(df_pca["PC1"], df_pca["PC2"], alpha=0.6)
        ax.set_xlabel("Principal Component 1")
        ax.set_ylabel("Principal Component 2")
        ax.set_title("PCA Scatter Plot")
        st.pyplot(fig)
    else:
        st.warning("Not enough numeric columns for PCA.")

    # ------------------------
    # Simple ML Model Training
    # ------------------------
    st.subheader("🤖 Logistic Regression Model")

    target_col = st.selectbox("Select Target Column", df_encoded.columns)
    X = df_encoded.drop(columns=[target_col])
    y = df_encoded[target_col]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    st.write("### ✅ Model Performance")
    st.write("**Accuracy:**", round(accuracy_score(y_test, y_pred), 3))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Confusion Matrix
    fig, ax = plt.subplots()
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Greens', ax=ax)
    ax.set_title("Confusion Matrix")
    st.pyplot(fig)

    # ------------------------
    # Download Processed Data
    # ------------------------
    st.subheader("⬇️ Download Processed Dataset")
    csv = df_encoded.to_csv(index=False).encode("utf-8")
    st.download_button("Download Processed CSV", csv, "processed_dataset.csv", "text/csv")

else:
    st.info("👆 Please upload a CSV file to begin analysis.")
