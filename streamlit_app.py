import streamlit as st
import joblib
import numpy as np
import os
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
    print("Models loaded successfully!")
    print(f"Catboost version used to train model: {catboost_hrc.get_param('versions')}")
    print(f"Catboost version in streamlit app: {catboost.__version__}")
except FileNotFoundError:
    st.error("Error: Model files not found. Ensure 'catboost_hrc.pkl' and 'catboost_kic.pkl' are in the same directory as this script.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Streamlit UI
st.title("Hot-Work Tool Steels Hardness & Toughness Prediction App")
st.write("Enter the composition and processing parameters to predict HRC and KIC.")

# Layout: Two columns for inputs
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Composition Inputs")
    C = float(st.number_input("Carbon (%)", min_value=0.0, max_value=2.0, step=0.01))
    Si = float(st.number_input("Silicon (%)", min_value=0.0, max_value=5.0, step=0.01))
    Mn = float(st.number_input("Manganese (%)", min_value=0.0, max_value=10.0, step=0.01))
    Cr = float(st.number_input("Chromium (%)", min_value=0.0, max_value=20.0, step=0.01))
    Mo = float(st.number_input("Molybdenum (%)", min_value=0.0, max_value=10.0, step=0.01))
    V = float(st.number_input("Vanadium (%)", min_value=0.0, max_value=5.0, step=0.01))
    Ni = float(st.number_input("Nickel (%)", min_value=0.0, max_value=10.0, step=0.01))
    W = float(st.number_input("Tungsten (%)", min_value=0.0, max_value=10.0, step=0.01))
    N = float(st.number_input("Nitrogen (%)", min_value=0.0, max_value=0.5, step=0.01))

with col2:
    st.subheader("Processing Inputs")
    Process = st.selectbox("Process Type", ['ESR', 'Conventional', 'PM'])
    Hardening = float(st.number_input("Hardening Temperature (°C)", min_value=500, max_value=1200, step=10))
    Tempering = float(st.number_input("Tempering Temperature (°C)", min_value=100, max_value=700, step=10))

# Button for prediction
if st.button("Predict HRC & KIC"):
    input_data = np.array([[C, Si, Mn, Cr, Mo, V, Ni, W, N, Process, Hardening, Tempering]])
    prediction_pool = Pool(data=input_data, cat_features=[9])
    hrc_prediction = catboost_hrc.predict(prediction_pool)[0]
    kic_prediction = catboost_kic.predict(prediction_pool)[0]
    
    # Layout for results and graphs
    result_col, graph_col = st.columns([1, 1])
    
    with result_col:
        st.subheader("Predicted Results")
        st.success(f"Predicted HRC: {hrc_prediction:.2f}")
        st.success(f"Predicted KIC: {kic_prediction:.2f}")
    
    with graph_col:
        st.subheader("Prediction Visualization")
        fig = go.Figure(data=[
            go.Bar(name='HRC', x=['Predicted'], y=[hrc_prediction]),
            go.Bar(name='KIC', x=['Predicted'], y=[kic_prediction])
        ])
        st.plotly_chart(fig)
    
    # Feature importance
    st.subheader("Feature Importance")
    fi_col1, fi_col2 = st.columns([1, 1])
    
    with fi_col1:
        hrc_feature_importance = catboost_hrc.get_feature_importance(prettified=True)
        fig_hrc_importance = px.bar(hrc_feature_importance, x='Feature Id', y='Importances', title='HRC Feature Importance')
        st.plotly_chart(fig_hrc_importance)
    
    with fi_col2:
        kic_feature_importance = catboost_kic.get_feature_importance(prettified=True)
        fig_kic_importance = px.bar(kic_feature_importance, x='Feature Id', y='Importances', title='KIC Feature Importance')
        st.plotly_chart(fig_kic_importance)
