import streamlit as st
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import altair as alt

# --------------------------------------------------
# Load the trained model pipeline.
# Update the file path to match your trained model file.
model = joblib.load(r"C:\Users\MaHaDi\Desktop\dashbord\best_model_pipeline.pkl")

# --------------------------------------------------
# Retrieve feature names from the scaler step of the pipeline.
# This relies on the scaler having the attribute "feature_names_in_".
# If not available, a fallback is used to create generic feature names.
try:
    if hasattr(model.named_steps["scaler"], "feature_names_in_"):
        feature_names = model.named_steps["scaler"].feature_names_in_
    else:
        # Fallback: Create generic feature names based on scaler's mean shape.
        n_features = model.named_steps["scaler"].mean_.shape[0]
        feature_names = [f"feature_{i}" for i in range(n_features)]
except Exception as e:
    st.error("Error extracting feature names from the model. "
             "Ensure that the model was trained using a DataFrame so that feature names are stored.")
    st.stop()

# --------------------------------------------------
# Define the quality label mapping.
# (These should match the label encoding used during training.)
quality_labels = {
    0: "Inefficient",
    1: "Acceptable",
    2: "Target",
    3: "Waste"
}

st.title("📈 Product Quality Prediction Dashboard")
st.markdown("Predict plastic product quality using the trained model.")

# --------------------------------------------------
# Sidebar - Input Features as Sliders
st.sidebar.header("🔧 Input Features")
input_data = {}
# The slider range from 0.0 to 100.0 with an initial value of 50.0 is assumed.
# Adjust these ranges as needed to match the distribution of your training data.
for feature in feature_names:
    input_data[feature] = st.sidebar.slider(feature, 0.0, 100.0, 50.0)

# Build a DataFrame for prediction ensuring the column order matches the training order.
input_df = pd.DataFrame([input_data])[feature_names]

# --------------------------------------------------
# Display the Input Data Table
st.subheader("🔍 Input Data")
st.dataframe(input_df)

# --------------------------------------------------
# Make Predictions
try:
    prediction = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
except Exception as e:
    st.error(f"Error during prediction: {e}")
    st.stop()

predicted_label = quality_labels.get(prediction, "Unknown")
st.success(f"✅ Predicted Quality: **{predicted_label}**")

# --------------------------------------------------
# Visual 1: Bar Chart of Prediction Probabilities
proba_df = pd.DataFrame({
    "Quality": list(quality_labels.values()),
    "Probability": proba
})
st.subheader("📊 Prediction Probabilities (Bar Chart)")
st.bar_chart(proba_df.set_index("Quality"))

# --------------------------------------------------
# Visual 2: Pie Chart of Prediction Probabilities
fig1, ax1 = plt.subplots()
ax1.pie(proba, labels=list(quality_labels.values()), autopct="%1.1f%%", startangle=90)
ax1.axis("equal")  # Ensures the pie chart is a circle.
st.subheader("📈 Prediction Probabilities (Pie Chart)")
st.pyplot(fig1)

# --------------------------------------------------
# Visual 3: Scrap vs Target Visual
# Here, we define scrap as the combined probability of 'Acceptable' (key=1) and 'Waste' (key=3)
scrap_prob = proba[1] + proba[3]  # Adjust these indices if your mapping differs.
target_prob = proba[2]  # Target probability

scrap_vs_target = pd.DataFrame({
    "Category": ["Target", "Scrap"],
    "Probability": [target_prob, scrap_prob]
})
st.subheader("📊 Target vs Scrap (Acceptable + Waste)")
st.bar_chart(scrap_vs_target.set_index("Category"))

# --------------------------------------------------
# Visual 4: Feature Importance Ranking
# Extract feature importances from the classifier in the pipeline.
try:
    importances = model.named_steps["clf"].feature_importances_
except AttributeError:
    st.error("The classifier does not provide feature_importances_.")
    importances = np.zeros(len(feature_names))

fi_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

st.subheader("⭐ Feature Importance Ranking")
st.table(fi_df)

# Altair horizontal bar chart for feature importances.
chart = alt.Chart(fi_df).mark_bar().encode(
    x=alt.X('Importance:Q', title="Importance"),
    y=alt.Y('Feature:N', sort='-x', title="Feature"),
    tooltip=['Feature', 'Importance']
).properties(
    width=600,
    height=400,
    title="Feature Importance"
)
st.altair_chart(chart, use_container_width=True)
