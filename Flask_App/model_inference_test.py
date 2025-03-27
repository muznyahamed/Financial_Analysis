import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
from pathlib import Path

# Set page config
st.set_page_config(
    page_title="Financial Prediction Model Test",
    page_icon="📊",
    layout="wide"
)

# Apply basic styling
st.markdown("""
<style>
    .header {
        color: #1E88E5;
        padding: 10px;
        text-align: center;
        margin-bottom: 20px;
    }
    .prediction-card {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 20px;
    }
    .prediction-value {
        font-size: 26px;
        font-weight: bold;
        color: #1E88E5;
    }
    .company-dipd {
        color: #1E88E5;
    }
    .company-rexp {
        color: #E53935;
    }
</style>
""", unsafe_allow_html=True)

def load_models():
    """Load the trained models and scalers"""
    try:
        # Get the models directory
        models_dir = Path("../data/processors/models")
        if not models_dir.exists():
            models_dir = Path("models")  # Try alternate location
        
        # Load DIPD model and scaler
        dipd_model = joblib.load(models_dir / 'dipd_model.joblib')
        dipd_scaler = joblib.load(models_dir / 'dipd_scaler.joblib')
        
        # Load REXP model and scaler
        rexp_model = joblib.load(models_dir / 'rexp_model.joblib')
        rexp_scaler = joblib.load(models_dir / 'rexp_scaler.joblib')
        
        return {
            'dipd': {'model': dipd_model, 'scaler': dipd_scaler},
            'rexp': {'model': rexp_model, 'scaler': rexp_scaler}
        }
    except Exception as e:
        st.error(f"Error loading models: {e}")
        st.info("Make sure model files exist in the correct location")
        return None

def make_predictions(input_data, models):
    """Make predictions using both models"""
    # Convert input to numpy array
    input_array = np.array([input_data])
    
    results = {}
    
    # DIPD prediction
    dipd_scaled = models['dipd']['scaler'].transform(input_array)
    dipd_prediction = models['dipd']['model'].predict(dipd_scaled)[0]
    results['dipd'] = dipd_prediction
    
    # REXP prediction
    rexp_scaled = models['rexp']['scaler'].transform(input_array)
    rexp_prediction = models['rexp']['model'].predict(rexp_scaled)[0]
    results['rexp'] = rexp_prediction
    
    return results

def main():
    st.markdown("<h1 class='header'>Financial Prediction Model Test</h1>", unsafe_allow_html=True)
    
    st.write("This application allows you to test the financial prediction models for DIPD and REXP companies.")
    
    # Load models
    models = load_models()
    if not models:
        st.stop()
    
    # Create input form
    st.subheader("Input Financial Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        revenue = st.number_input("Revenue", min_value=0, value=10000000, step=100000, 
                                 help="Total revenue for the period")
        cost_of_sales = st.number_input("Cost of Sales", min_value=0, value=8000000, step=100000,
                                       help="Total cost of goods sold")
        gross_profit = st.number_input("Gross Profit", min_value=0, value=2000000, step=10000,
                                      help="Revenue minus cost of sales")
    
    with col2:
        operating_expenses = st.number_input("Operating Expenses", min_value=0, value=1000000, step=10000,
                                           help="Total operating expenses")
        operating_income = st.number_input("Operating Income", min_value=-10000000, value=1000000, step=10000,
                                          help="Gross profit minus operating expenses")
    
    # Calculate missing values automatically
    if st.checkbox("Auto-calculate gross profit", value=True):
        gross_profit = revenue - cost_of_sales
    
    if st.checkbox("Auto-calculate operating income", value=True):
        operating_income = gross_profit - operating_expenses
    
    # Create feature array
    input_data = [revenue, cost_of_sales, gross_profit, operating_expenses, operating_income]
    
    # Make predictions when user clicks button
    if st.button("Make Predictions"):
        with st.spinner("Calculating predictions..."):
            predictions = make_predictions(input_data, models)
            
            # Display predictions
            st.subheader("Prediction Results")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='company-dipd'>DIPD Inc. Prediction</h3>", unsafe_allow_html=True)
                st.markdown(f"<p class='prediction-value'>{predictions['dipd']:,.2f}</p>", unsafe_allow_html=True)
                st.markdown("Predicted Net Income")
                st.markdown("</div>", unsafe_allow_html=True)
            
            with col2:
                st.markdown("<div class='prediction-card'>", unsafe_allow_html=True)
                st.markdown("<h3 class='company-rexp'>REXP Corp. Prediction</h3>", unsafe_allow_html=True)
                st.markdown(f"<p class='prediction-value'>{predictions['rexp']:,.2f}</p>", unsafe_allow_html=True)
                st.markdown("Predicted Net Income")
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Display comparison
            difference = abs(predictions['dipd'] - predictions['rexp'])
            percentage_diff = (difference / ((predictions['dipd'] + predictions['rexp'])/2)) * 100
            
            st.subheader("Comparison Analysis")
            st.write(f"Absolute difference between predictions: {difference:,.2f}")
            st.write(f"Percentage difference: {percentage_diff:.2f}%")
            
            # Create a bar chart for visualization
            fig, ax = plt.subplots(figsize=(10, 6))
            companies = ['DIPD Inc.', 'REXP Corp.']
            values = [predictions['dipd'], predictions['rexp']]
            colors = ['#1E88E5', '#E53935']
            
            ax.bar(companies, values, color=colors)
            ax.set_ylabel('Predicted Net Income')
            ax.set_title('Prediction Comparison')
            
            # Format y-axis with commas
            import matplotlib.ticker as mtick
            ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x, p: format(int(x), ',')))
            
            st.pyplot(fig)
            
            # Analysis of the results
            st.subheader("Prediction Analysis")
            
            if predictions['dipd'] > predictions['rexp']:
                st.write("DIPD Inc. is predicted to have higher net income compared to REXP Corp. for the given financial metrics.")
            elif predictions['rexp'] > predictions['dipd']:
                st.write("REXP Corp. is predicted to have higher net income compared to DIPD Inc. for the given financial metrics.")
            else:
                st.write("Both companies are predicted to have similar net income for the given financial metrics.")
            
            if percentage_diff > 50:
                st.warning("There's a significant difference between the predictions of the two companies. This could be due to differences in business models, operational efficiency, or market conditions.")
            elif percentage_diff < 10:
                st.success("The predictions are relatively close, suggesting similar financial performance for both companies given these metrics.")

if __name__ == "__main__":
    main() 