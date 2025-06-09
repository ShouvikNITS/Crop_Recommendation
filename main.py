# main.py

from crop_recommendation_system import CropRecommendationSystem
import streamlit as st


def predict_crop(location, n, p, k, ph):
    # Full path to your CSV file
    csv_file = "C:/Users/LENOVO/Desktop/College Projects/ML and DL Projects/Crop Recommendation Model/Crop_recommendation.csv"
    
    # Initialize system
    system = CropRecommendationSystem(csv_file)
    system.setup_system(train_new_model=True)
    
    # Get recommendations
    return system.get_recommendations(location, n, p, k, ph)


def main():
    st.title('🌾 Crop Recommendation System')

    location = st.text_input('📍 Enter your location:')

    # Optional soil input
    st.markdown("### (Optional) Soil Parameters")
    n = st.number_input("Nitrogen (N)", min_value=0.0, max_value=500.0, step=1.0, value=0.0)
    p = st.number_input("Phosphorus (P)", min_value=0.0, max_value=500.0, step=1.0, value=0.0)
    k = st.number_input("Potassium (K)", min_value=0.0, max_value=500.0, step=1.0, value=0.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, step=0.1, value=7.0)

    if st.button('🌱 Recommend Crops'):
        recommendations = predict_crop(location, n, p, k, ph)
        
        if recommendations:
            st.success(f"Top 3 Crop Recommendations for {location}:")
            for i, (crop, probability) in enumerate(recommendations, 1):
                confidence = "High" if probability > 0.7 else "Medium" if probability > 0.4 else "Low"
                st.markdown(f"**{i}. {crop}**  \nProbability: {probability:.2f} ({confidence} confidence)")
        else:
            st.error("No recommendation found for this location.")

if __name__ == "__main__":
    main()
