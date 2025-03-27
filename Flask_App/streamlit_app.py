import streamlit as st
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import requests
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
# Import the components module
from components import (
    load_all_companies_data, 
    display_company_metrics, 
    render_overview_tab,
    render_revenue_analysis_tab,
    render_margin_analysis_tab,
    render_quarterly_details_tab,
    render_company_comparison,
    apply_custom_css,
    apply_chat_css,
    initialize_agent
)

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Multi-Company Financial Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS
apply_custom_css()

# Company configuration
company_configs = {
    "DIPD": {
        "name": "DIPD Inc.",
        "file_path": "Data/dipd/dipd_quarterly.json",
        "color": "blue"
    },
    "REXP": {
        "name": "REXP Corp.",
        "file_path": "Data/rexp/data_quartile.json",
        "color": "red"
    }
}

# Load all companies data
all_companies_data = load_all_companies_data(company_configs)

# Initialize app modes in session state if they don't exist
if 'chat_mode' not in st.session_state:
    st.session_state.chat_mode = False
    
if 'prediction_mode' not in st.session_state:
    st.session_state.prediction_mode = False

# Initialize chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize agent if it doesn't exist
if 'agent' not in st.session_state and not st.session_state.chat_mode:
    st.session_state.agent = None

# Function to toggle chat mode
def toggle_chat():
    st.session_state.chat_mode = not st.session_state.chat_mode
    # When entering chat mode, turn off prediction mode
    if st.session_state.chat_mode:
        st.session_state.prediction_mode = False
        if st.session_state.agent is None:
            with st.spinner("Initializing AI assistant..."):
                st.session_state.agent = initialize_agent(all_companies_data, company_configs)

# Function to toggle prediction mode
def toggle_prediction():
    st.session_state.prediction_mode = not st.session_state.prediction_mode
    # When entering prediction mode, turn off chat mode
    if st.session_state.prediction_mode:
        st.session_state.chat_mode = False

# Function to load ML models for prediction
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

# Function to make predictions
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

# Function to render the prediction UI
def render_prediction_ui():
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
        .input-note {
            background-color: #e3f2fd;
            border-left: 4px solid #1E88E5;
            padding: 10px;
            margin-bottom: 15px;
            border-radius: 0 4px 4px 0;
        }
    </style>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 class='header'>Financial Prediction Model</h1>", unsafe_allow_html=True)
    
    st.write("This section allows you to test the financial prediction models for DIPD and REXP companies.")
    
    # Add note about using last quarter data
    st.markdown("""
    <div class="input-note">
        <b>Important:</b> Please input financial data from the <b>last previous quarter</b> for accurate predictions. 
        The model uses these values to forecast the next quarter's net income.
    </div>
    """, unsafe_allow_html=True)
    
    # Load models
    models = load_models()
    if not models:
        st.stop()
    
    # Create input form
    st.subheader("Input Previous Quarter's Financial Data")
    
    col1, col2 = st.columns(2)
    
    with col1:
        revenue = st.number_input("Previous Quarter Revenue", min_value=0, value=10000000, step=100000, 
                                 help="Total revenue from the previous quarter")
        cost_of_sales = st.number_input("Previous Quarter Cost of Sales", min_value=0, value=8000000, step=100000,
                                       help="Total cost of goods sold from the previous quarter")
        gross_profit = st.number_input("Previous Quarter Gross Profit", min_value=0, value=2000000, step=10000,
                                      help="Revenue minus cost of sales from the previous quarter")
    
    with col2:
        operating_expenses = st.number_input("Previous Quarter Operating Expenses", min_value=0, value=1000000, step=10000,
                                           help="Total operating expenses from the previous quarter")
        operating_income = st.number_input("Previous Quarter Operating Income", min_value=-10000000, value=1000000, step=10000,
                                          help="Gross profit minus operating expenses from the previous quarter")
    
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
            st.subheader("Next Quarter Prediction Results")
            
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

# App header
if not st.session_state.chat_mode and not st.session_state.prediction_mode:
    st.title("📊 Multi-Company Financial Performance Dashboard")
    st.markdown("Interactive dashboard for analyzing and comparing quarterly financial performance")
elif st.session_state.chat_mode:
    # Apply chat-specific CSS
    apply_chat_css()
# No header needed for prediction mode - it has its own

# App description and user guide in sidebar
st.sidebar.header("App Overview")

# Mode navigation buttons in sidebar
chat_btn_label = "Switch to Chat Assistant" if not st.session_state.chat_mode else "Return to Dashboard"
if st.sidebar.button(chat_btn_label, use_container_width=True):
    toggle_chat()

# Add prediction button below chat button
prediction_btn_label = "Switch to Prediction Model" if not st.session_state.prediction_mode else "Return to Dashboard"
if st.sidebar.button(prediction_btn_label, use_container_width=True):
    toggle_prediction()

# Main content based on current mode
if st.session_state.prediction_mode:
    # Render prediction UI when in prediction mode
    render_prediction_ui()
elif st.session_state.chat_mode:
    # Chat mode UI
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OpenAI API key not found. Please add your API key to the .env file.")
        st.info("Example format: OPENAI_API_KEY=your_key_here")
        st.stop()
    
    # Create a container for the entire chat interface
    chat_container = st.container()
    
    # Create a container for the info sidebar
    with st.sidebar:
        st.subheader("About this Assistant")
        st.markdown("""
        This AI assistant can:
        - Analyze financial data from the dashboard
        - Provide insights on company performance
        - Compare metrics between companies
        - Offer recommendations based on trends
        - Answer general questions about Sri Lanka's business environment
        """)
        
        st.subheader("Available Companies")
        for company_id, config in company_configs.items():
            if all_companies_data[company_id] is not None:
                df = all_companies_data[company_id]
                st.markdown(f"""
                **{config['name']}**  
                Data from: {df['date'].min().strftime('%b %Y')} to {df['date'].max().strftime('%b %Y')}
                """)
        
        with st.expander("Example Questions", expanded=True):
            st.markdown("""
            - What is DIPD's revenue trend over the last year?
            - Compare the gross margins of DIPD and REXP
            - What are REXP's key financial strengths and weaknesses?
            - How has DIPD's operating efficiency changed over time?
            - What's the business outlook for technology companies in Sri Lanka?
            - What are the main economic challenges facing businesses in Sri Lanka?
            """)
        
        # Add option to reset chat
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.agent.reset_memory()
            st.rerun()
    
    # Create the main chat interface
    with chat_container:
        # Create a layout with three rows:
        # 1. Header
        # 2. Messages (takes most space)
        # 3. Input field (at bottom)
        
        # 1. Header row
        st.markdown("""
        <div class="finance-header">
            <h1>💬 Financial AI Assistant</h1>
            <p>Ask questions about financial performance, get analysis and recommendations</p>
        </div>
        """, unsafe_allow_html=True)
        
        # 2. Messages container (will take most of the vertical space)
        messages_container = st.container()
        
        # 3. Input container (will be at the bottom)
        input_container = st.container()
        
        # Create a container for styling the bottom input area
        with input_container:
            st.markdown('<div class="chat-input-area">', unsafe_allow_html=True)
            prompt = st.chat_input("Ask me about financial data or Sri Lankan business insights...")
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Display messages in the messages container
        with messages_container:
            st.markdown('<div class="chat-messages-container">', unsafe_allow_html=True)
            # Display chat messages from history
            for message in st.session_state.chat_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Handle new user input
        if prompt:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with messages_container:
                with st.chat_message("user"):
                    st.markdown(prompt)
            
            # Display assistant response
            with messages_container:
                with st.chat_message("assistant"):
                    with st.spinner("Analyzing financial data..."):
                        message_placeholder = st.empty()
                        try:
                            response = st.session_state.agent.run(prompt)
                            message_placeholder.markdown(response)
                            # Add assistant response to chat history
                            st.session_state.chat_history.append({"role": "assistant", "content": response})
                        except Exception as e:
                            error_msg = f"Error: {str(e)}"
                            message_placeholder.error(error_msg)
                            st.session_state.chat_history.append({"role": "assistant", "content": error_msg})
            
            # Rerun to refresh the UI correctly
            st.rerun()
else:
    # Dashboard mode UI
    # Dashboard mode description
    st.sidebar.markdown("""
    ### Multi-Company Financial Dashboard

    This dashboard provides an interactive platform to analyze and compare the quarterly financial performance of multiple companies. 

    #### Key Features:
    - **Company Selection**: Choose a company to view its financial data.
    - **Date Range Filter**: Select a specific date range to filter the data.
    - **Year Filter**: Choose specific years to focus on.
    - **Company Comparison**: Compare financial metrics between two companies.

    #### Key Metrics:
    - **Revenue**: Total income generated from sales.
    - **Net Income**: Profit after all expenses have been deducted from revenue.
    - **Gross Margin**: Percentage of revenue remaining after deducting the cost of goods sold.
    - **Operating Margin**: Percentage of revenue remaining after deducting operating expenses.
    - **Net Margin**: Percentage of revenue remaining after all expenses, including taxes and interest, have been deducted.

    #### User Guide:
    1. **Select a Company**: Use the sidebar to choose a company from the available options.
    2. **Apply Filters**: Adjust the date range and select specific years to refine the data.
    3. **Enable Comparison**: Check the box to compare the selected company with another available company.
    4. **Explore Tabs**: Navigate through the tabs to view different analyses such as revenue trends, margin analysis, and quarterly details.
    5. **View KPIs**: The main page displays key performance indicators for the most recent quarter.

    Explore the dashboard to gain insights into the financial health and performance trends of the companies.
    """)

    # Company selector
    st.sidebar.header("Company Selection")
    company_options = [config["name"] for company_id, config in company_configs.items() 
                      if all_companies_data[company_id] is not None]
    company_ids = [company_id for company_id, config in company_configs.items() 
                  if all_companies_data[company_id] is not None]

    selected_company_name = st.sidebar.selectbox(
        "Select Company",
        options=company_options,
        index=0
    )

    # Get the company ID from the selected name
    selected_company_id = company_ids[company_options.index(selected_company_name)]
    df = all_companies_data[selected_company_id]

    # Display company info banner
    st.markdown(f"""
    <div class="company-selector">
        <h3>📈 {selected_company_name}</h3>
        <p>Viewing financial data from {df['date'].min().strftime('%b %Y')} to {df['date'].max().strftime('%b %Y')}</p>
    </div>
    """, unsafe_allow_html=True)

    # Sidebar filters
    st.sidebar.header("Filters")

    # Date range filter
    date_range = st.sidebar.date_input(
        "Date Range",
        [df["date"].min(), df["date"].max()],
        min_value=df["date"].min(),
        max_value=df["date"].max()
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered_df = df[(df["date"] >= pd.Timestamp(start_date)) & 
                         (df["date"] <= pd.Timestamp(end_date))]
    else:
        filtered_df = df

    # Year filter
    years = filtered_df["year"].unique().tolist()
    selected_years = st.sidebar.multiselect("Select Years", options=years, default=years[-2:])

    if selected_years:
        year_filtered_df = filtered_df[filtered_df["year"].isin(selected_years)]
    else:
        year_filtered_df = filtered_df

    # Add company comparison option
    st.sidebar.header("Company Comparison")
    enable_comparison = st.sidebar.checkbox("Enable Company Comparison", value=False)
    comparison_company_id = None
    comparison_company_name = None
    comparison_df = None

    if enable_comparison:
        comparison_company_options = [config["name"] for company_id, config in company_configs.items() 
                                    if company_id != selected_company_id and all_companies_data[company_id] is not None]
        comparison_company_ids = [company_id for company_id, config in company_configs.items() 
                                if company_id != selected_company_id and all_companies_data[company_id] is not None]
        
        if comparison_company_options:
            comparison_company_name = st.sidebar.selectbox(
                "Compare with",
                options=comparison_company_options,
                index=0
            )
            comparison_company_id = comparison_company_ids[comparison_company_options.index(comparison_company_name)]
            comparison_df = all_companies_data[comparison_company_id]
        else:
            enable_comparison = False
            st.sidebar.warning("No other companies available for comparison")

    # Show KPIs for the most recent quarter
    display_company_metrics(year_filtered_df)

    # Create tabs for different analyses
    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "Revenue Analysis", "Margin Analysis", "Quarterly Details"])

    with tab1:
        render_overview_tab(
            year_filtered_df, 
            selected_company_id, 
            selected_company_name, 
            enable_comparison, 
            comparison_df, 
            comparison_company_id, 
            comparison_company_name, 
            company_configs
        )

    with tab2:
        render_revenue_analysis_tab(
            year_filtered_df, 
            selected_company_id, 
            selected_company_name, 
            selected_years, 
            company_configs
        )

    with tab3:
        render_margin_analysis_tab(
            year_filtered_df, 
            selected_company_name, 
            enable_comparison, 
            comparison_df, 
            comparison_company_name, 
            selected_years
        )

    with tab4:
        render_quarterly_details_tab(year_filtered_df)

    # Add a comparative analysis tab if multiple companies are available
    if enable_comparison:
        render_company_comparison(
            year_filtered_df, 
            comparison_df, 
            selected_company_name, 
            comparison_company_name, 
            selected_company_id, 
            comparison_company_id, 
            company_configs, 
            selected_years
        )

# Footer
st.markdown("---")
st.caption("Multi-Company Financial Dashboard powered by Streamlit • Data refreshed quarterly")