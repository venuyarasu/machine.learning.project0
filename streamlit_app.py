import streamlit as st
import joblib
import numpy as np
from catboost import Pool
import catboost
import plotly.graph_objects as go
import plotly.express as px

# Model file paths (relative to the script location)
hrc_model_path = 'catboost_hrc.pkl'
kic_model_path = 'catboost_kic.pkl'

# Load trained models with error handling
try:
    catboost_hrc = joblib.load(hrc_model_path)
    catboost_kic = joblib.load(kic_model_path)
    st.write("Models loaded successfully!")
    st.write(f"CatBoost version used to train model: {catboost_hrc.get_param('iterations')}")
    st.write(f"CatBoost version in Streamlit app: {catboost.__version__}")

except FileNotFoundError:
    st.error("Error: Model files not found. Ensure 'catboost_hrc.pkl' and 'catboost_kic.pkl' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Streamlit UI
st.title("Hot-Work Tool Steels Hardness & Toughness Prediction App")
st.write("Enter the composition and processing parameters to predict HRC and KIC.")

# User Inputs (with explicit type conversions)
C = float(st.number_input("Carbon (%)", min_value=0.0, max_value=2.0, step=0.01))
Si = float(st.number_input("Silicon (%)", min_value=0.0, max_value=5.0, step=0.01))
Mn = float(st.number_input("Manganese (%)", min_value=0.0, max_value=10.0, step=0.01))
Cr = float(st.number_input("Chromium (%)", min_value=0.0, max_value=20.0, step=0.01))
Mo = float(st.number_input("Molybdenum (%)", min_value=0.0, max_value=10.0, step=0.01))
V = float(st.number_input("Vanadium (%)", min_value=0.0, max_value=5.0, step=0.01))
Ni = float(st.number_input("Nickel (%)", min_value=0.0, max_value=10.0, step=0.01))
W = float(st.number_input("Tungsten (%)", min_value=0.0, max_value=10.0, step=0.01))
N = float(st.number_input("Nitrogen (%)", min_value=0.0, max_value=0.5, step=0.01))
Hardening = float(st.number_input("Hardening Temperature (°C)", min_value=500, max_value=1200, step=10))
Tempering = float(st.number_input("Tempering Temperature (°C)", min_value=100, max_value=700, step=10))
Process = st.selectbox("Process Type", ['ESR', 'Conventional', 'PM'])

# Process Encoding
process_mapping = {'ESR': 0, 'Conventional': 1, 'PM': 2}
process_encoded = process_mapping[Process]

# Prediction button
if st.button("Predict HRC & KIC"):
    # Prepare input for model
    input_data = np.array([[C, Si, Mn, Cr, Mo, V, Ni, W, N, process_encoded, Hardening, Tempering]])
    
    # Validate shape
    assert input_data.shape == (1, 12), f"Expected shape (1, 12), but got {input_data.shape}"

    # Create a CatBoost Pool object
    prediction_pool = Pool(data=input_data, cat_features=[9])

    # Predict
    hrc_prediction = catboost_hrc.predict(prediction_pool)[0]
    kic_prediction = catboost_kic.predict(prediction_pool)[0]

    # Display results
    st.success(f"Predicted HRC: {hrc_prediction:.2f}")
    st.success(f"Predicted KIC: {kic_prediction:.2f}")

    # Visualization of results
    fig = go.Figure(data=[
        go.Bar(name='HRC', x=['Predicted'], y=[hrc_prediction]),
        go.Bar(name='KIC', x=['Predicted'], y=[kic_prediction])
    ])
    st.plotly_chart(fig)

    # Feature Importance
    hrc_feature_importance = catboost_hrc.get_feature_importance(prettified=True)
    hrc_feature_importance.columns = ['Feature', 'Importance']
    fig_hrc_importance = px.bar(hrc_feature_importance, x='Feature', y='Importance')
    st.subheader("HRC Feature Importance")
    st.plotly_chart(fig_hrc_importance)

    kic_feature_importance = catboost_kic.get_feature_importance(prettified=True)
    kic_feature_importance.columns = ['Feature', 'Importance']
    fig_kic_importance = px.bar(kic_feature_importance, x='Feature', y='Importance')
    st.subheader("KIC Feature Importance")
    st.plotly_chart(fig_kic_importance)
