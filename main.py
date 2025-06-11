# main.py

import streamlit as st
from crop_recommendation_system import CropRecommendationSystem
import time


def predict_crop(location, n, p, k, ph):
    csv_file = "Crop_recommendation.csv"
    system = CropRecommendationSystem(csv_file)
    system.setup_system(train_new_model=True)
    return system.get_recommendations(location, n, p, k, ph)


def main():
    # Set page config
    st.set_page_config(
        page_title="Crop Recommendation System",
        page_icon="🌾",
        layout="centered",
    )

    # Header section
    st.markdown("<h1 style='text-align: center; color: green;'>🌿 Smart Crop Recommendation System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center;'>Empowering farmers with data-driven suggestions for better yields 🚜</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Sidebar inputs
    st.sidebar.header("📍 Location & Soil Inputs")
    location = st.sidebar.text_input("Enter your location", placeholder="e.g., Delhi")

    st.sidebar.markdown("### 🌱 Soil Nutrients (Optional)")
    n = st.sidebar.number_input("Nitrogen (N)", min_value=0.0, max_value=500.0, step=1.0, value=0.0)
    p = st.sidebar.number_input("Phosphorus (P)", min_value=0.0, max_value=500.0, step=1.0, value=0.0)
    k = st.sidebar.number_input("Potassium (K)", min_value=0.0, max_value=500.0, step=1.0, value=0.0)
    ph = st.sidebar.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1, value=7.0)

    # Main button
    if st.sidebar.button("🌱 Recommend Crops"):
        if not location:
            st.error("⚠️ Please enter a location before proceeding.")
            return

        with st.spinner("Analyzing soil and climate conditions..."):
            time.sleep(1.5)
            recommendations = predict_crop(location, n, p, k, ph)

        st.markdown("---")
        st.subheader(f"📊 Recommended Crops for: `{location}`")

        if recommendations:
            for i, (crop, probability) in enumerate(recommendations, 1):
                confidence = "High ✅" if probability > 0.7 else "Medium ⚠️" if probability > 0.4 else "Low ❌"
                st.markdown(f"""
                    <div style="background-color:#e8f5e9;padding:10px;border-radius:10px;margin-bottom:10px">
                        <h4>{i}. 🌾 {crop}</h4>
                        <b>Probability:</b> {probability:.2f} <br>
                        <b>Confidence:</b> {confidence}
                    </div>
                """, unsafe_allow_html=True)
        else:
            st.error("❌ Sorry, we couldn't generate recommendations for this location.")

    # Expandable section
    with st.expander("ℹ️ How this works"):
        st.markdown("""
        This app uses a machine learning model trained on soil nutrients and environmental conditions to recommend the most suitable crops for a location.  
        You can optionally input NPK values and pH, or leave them as default.
        """)

    st.markdown("---")
    st.caption("© 2025 GreenInnovators")

if __name__ == "__main__":
    main()
