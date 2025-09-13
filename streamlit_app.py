# app/streamlit_app.py

import streamlit as st
import pandas as pd
import os
import sys
import plotly.express as px

# -----------------------------
# Page config (must be first)
# -----------------------------
st.set_page_config(
    page_title="🌾 Sustainable Crop Recommendation",
    layout="wide",
    page_icon="🌱"
)

# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.crop_predictor import load_models, predict_crop

# -----------------------------
# Load models & encoders
# -----------------------------
@st.cache_resource
def get_models():
    return load_models()

MODELS = get_models()

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("🌍 Sustainable Crop Recommendation System")
st.markdown("""
AI-powered tool to recommend the **best crop** based on soil & weather conditions.  
Models: **Random Forest**, **SVM**.
""")

# -----------------------------
# Sidebar inputs (interactive sliders)
# -----------------------------
st.sidebar.header("📥 Enter Farm Data")

input_features = {
    "n": st.sidebar.slider("Nitrogen (N)", 0, 150, 40),
    "p": st.sidebar.slider("Phosphorus (P)", 0, 150, 50),
    "k": st.sidebar.slider("Potassium (K)", 0, 200, 40),
    "temperature": st.sidebar.slider("Temperature (°C)", 0.0, 50.0, 25.0),
    "humidity": st.sidebar.slider("Humidity (%)", 0.0, 100.0, 80.0),
    "ph": st.sidebar.slider("Soil pH", 0.0, 14.0, 6.5),
    "rainfall": st.sidebar.slider("Rainfall (mm)", 0.0, 400.0, 200.0),
}

st.sidebar.markdown("### 🌱 Input Summary")
st.sidebar.write(input_features)

# -----------------------------
# Predict button / dynamic update
# -----------------------------
if st.sidebar.button("🔍 Predict Crop"):

    X_input = pd.DataFrame([input_features])

    # Function to get top 3 crops with probabilities
    def get_top3(model, X):
        try:
            probs = model.predict_proba(X)[0]
            classes = MODELS["encoders"]["label"].classes_
            df = pd.DataFrame({"crop": classes, "prob": probs})
            return df.sort_values("prob", ascending=False).head(3)
        except AttributeError:
            pred = model.predict(X)[0]
            return pd.DataFrame({"crop": [pred], "prob": [1.0]})

    # Get top 3 for each model
    rf_top3 = get_top3(MODELS["random_forest"], X_input)
    svm_top3 = get_top3(MODELS["svm"], X_input)

    # -----------------------------
    # Display tables
    # -----------------------------
    st.subheader("✅ Recommended Crops (Top 3)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🌾 Random Forest")
        st.table(rf_top3)

    with col2:
        st.markdown("### 🌱 SVM")
        st.table(svm_top3)

    # -----------------------------
    # Combined interactive bar chart
    # -----------------------------
    combined_df = pd.DataFrame({
        "crop": list(rf_top3["crop"]) + list(svm_top3["crop"]),
        "probability": list(rf_top3["prob"]) + list(svm_top3["prob"]),
        "model": ["Random Forest"]*3 + ["SVM"]*3
    })

    st.subheader("📊 Crop Probability Comparison")
    fig = px.bar(
        combined_df,
        x="probability",
        y="crop",
        color="model",
        orientation="h",
        barmode="group",
        text="probability",
        height=400
    )
    fig.update_layout(
        xaxis_title="Probability",
        yaxis_title="Crop",
        yaxis={'categoryorder':'total ascending'},
        xaxis=dict(range=[0,1])
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👈 Adjust sliders and click **Predict Crop** to see recommendations.")
