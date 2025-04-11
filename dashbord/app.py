import streamlit as st
import pandas as pd
import joblib
import numpy as np
from PIL import Image
from scipy.stats import zscore

# Streamlit page setup
st.set_page_config(page_title="Product Quality Prediction – Gradient Boost Model", layout="wide")
st.title("🎯 Product Quality Prediction – Gradient Boost Model")

# Class mapping
quality_mapping = {
    1: '🗑️ Waste',
    2: '🟡 Acceptable',
    3: '✅ Target Product',
    4: '⚠️ Inefficient'
}

# Load model and dataset
@st.cache_resource
def load_model():
    return joblib.load(r"C:\Users\MaHaDi\Desktop\dashbord\gradient_boost_model.pkl")

@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\MaHaDi\Desktop\dashbord\CW_Dataset_4123490.csv")
    df.dropna(inplace=True)

    # Remove outliers using Z-score
    df_numeric = df.select_dtypes(include=[np.number])
    z_scores = np.abs(zscore(df_numeric))
    df_clean = df[(z_scores < 3).all(axis=1)]
    return df_clean

model = load_model()
df = load_data()

# Ensure target column exists
if "quality" not in df.columns:
    st.error("❌ 'quality' column not found.")
    st.stop()

# Get input feature names
feature_names = df.drop("quality", axis=1).columns.tolist()

# Sidebar for inputs
st.sidebar.header("🔧 Input Parameters")
user_input = {}
for feature in feature_names:
    min_val = float(df[feature].min())
    max_val = float(df[feature].max())
    mean_val = float(df[feature].mean())
    user_input[feature] = st.sidebar.slider(
        label=feature,
        min_value=min_val,
        max_value=max_val,
        value=mean_val
    )

# Convert user input to DataFrame
input_df = pd.DataFrame([user_input])

# Predict on button click
if st.button("🔍 Predict Quality Class"):
    prediction = model.predict(input_df)[0]  # This will be an integer: 1, 2, 3, or 4
    predicted_label = quality_mapping.get(int(prediction), "❓ Unknown Class")
    st.success(f"📦 Predicted Product Quality Class: **{predicted_label}**")

# Optional: preview dataset
with st.expander("📄 View Sample of Dataset"):
    st.dataframe(df.head(10))

# Visualizations
st.subheader("📊 Model Visualizations – Gradient Boost")

try:
    cm_img = Image.open("catboost (after tuning).png")
    st.image(cm_img, caption="Confusion Matrix – Gradient Boost")
except:
    st.info("📷 Confusion matrix image not found.")

try:
    roc_img = Image.open("catboost ROC.png")
    st.image(roc_img, caption="ROC Curve – Gradient Boost")
except:
    st.info("📷 ROC curve image not found.")

try:
    fi_img = Image.open("catboost feature importance.png")
    st.image(fi_img, caption="Top 10 Feature Importances")
except:
    st.info("📷 Feature importance image not found.")
