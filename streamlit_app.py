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
except FileNotFoundError:
    st.error("Error: Model files not found. Ensure 'catboost_hrc.pkl' and 'catboost_kic.pkl' are in the same directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Streamlit UI
st.title("Hot-Work Tool Steels Hardness & Toughness Prediction App")

# Layout: Two columns for inputs
col1, col2 = st.columns(2)

with col1:
    st.subheader("Composition Inputs")
    C = st.number_input("Carbon (%)", min_value=0.0, max_value=2.0, step=0.01)
    Si = st.number_input("Silicon (%)", min_value=0.0, max_value=5.0, step=0.01)
    Mn = st.number_input("Manganese (%)", min_value=0.0, max_value=10.0, step=0.01)
    Cr = st.number_input("Chromium (%)", min_value=0.0, max_value=20.0, step=0.01)
    Mo = st.number_input("Molybdenum (%)", min_value=0.0, max_value=10.0, step=0.01)
    V = st.number_input("Vanadium (%)", min_value=0.0, max_value=5.0, step=0.01)
    Ni = st.number_input("Nickel (%)", min_value=0.0, max_value=10.0, step=0.01)
    W = st.number_input("Tungsten (%)", min_value=0.0, max_value=10.0, step=0.01)
    N = st.number_input("Nitrogen (%)", min_value=0.0, max_value=0.5, step=0.01)

with col2:
    st.subheader("Processing Inputs")
    Process = st.selectbox("Process Type", ['ESR', 'Conventional', 'PM'])
    Hardening = st.number_input("Hardening Temperature (°C)", min_value=500, max_value=1200, step=10)
    Tempering = st.number_input("Tempering Temperature (°C)", min_value=100, max_value=700, step=10)

if st.button("Predict HRC & KIC"):
    input_data = np.array([[C, Si, Mn, Cr, Mo, V, Ni, W, N, Process, Hardening, Tempering]])
    prediction_pool = Pool(data=input_data, cat_features=[9])
    hrc_prediction = catboost_hrc.predict(prediction_pool)[0]
    kic_prediction = catboost_kic.predict(prediction_pool)[0]
    
    # Results and graphs layout
    result_col, graph_col = st.columns([1, 1])
    
    with result_col:
        st.subheader("Predicted Results")
        st.write(f"**HRC:** {hrc_prediction:.2f}")
        st.write(f"**KIC:** {kic_prediction:.2f}")
    
    with graph_col:
        st.subheader("Prediction Visualization")
        fig = go.Figure(data=[
            go.Bar(name='HRC', x=['Predicted'], y=[hrc_prediction]),
            go.Bar(name='KIC', x=['Predicted'], y=[kic_prediction])
        ])
        st.plotly_chart(fig, use_container_width=True)
    
    # Feature importance in expandable sections
    with st.expander("Feature Importance"):
        fi_col1, fi_col2 = st.columns(2)
        
        with fi_col1:
            hrc_feature_importance = catboost_hrc.get_feature_importance(prettified=True)
            fig_hrc_importance = px.bar(hrc_feature_importance, x='Feature Id', y='Importances', title='HRC Importance')
            st.plotly_chart(fig_hrc_importance, use_container_width=True)
        
        with fi_col2:
            kic_feature_importance = catboost_kic.get_feature_importance(prettified=True)
            fig_kic_importance = px.bar(kic_feature_importance, x='Feature Id', y='Importances', title='KIC Importance')
            st.plotly_chart(fig_kic_importance, use_container_width=True)
