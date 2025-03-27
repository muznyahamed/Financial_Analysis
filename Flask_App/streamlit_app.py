import streamlit as st
import pandas as pd
from datetime import datetime
import os
from dotenv import load_dotenv
import requests
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

# Initialize chat mode in session state if it doesn't exist
if 'chat_mode' not in st.session_state:
    st.session_state.chat_mode = False

# Initialize chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

# Initialize agent if it doesn't exist
if 'agent' not in st.session_state and not st.session_state.chat_mode:
    st.session_state.agent = None

# Function to toggle chat mode
def toggle_chat():
    st.session_state.chat_mode = not st.session_state.chat_mode
    if st.session_state.chat_mode and st.session_state.agent is None:
        with st.spinner("Initializing AI assistant..."):
            st.session_state.agent = initialize_agent(all_companies_data, company_configs)

# App header
if not st.session_state.chat_mode:
    st.title("📊 Multi-Company Financial Performance Dashboard")
    st.markdown("Interactive dashboard for analyzing and comparing quarterly financial performance")
else:
    # Apply chat-specific CSS
    apply_chat_css()
    st.markdown("""
    <div class="finance-header">
        <h1>💬 Financial AI Assistant</h1>
        <p>Ask questions about financial performance, get analysis and recommendations</p>
    </div>
    """, unsafe_allow_html=True)

# App description and user guide in sidebar
st.sidebar.header("App Overview")

# Chat toggle button in sidebar
chat_btn_label = "Switch to Chat Assistant" if not st.session_state.chat_mode else "Return to Dashboard"
if st.sidebar.button(chat_btn_label, use_container_width=True):
    toggle_chat()

if not st.session_state.chat_mode:
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
else:
    # Chat mode UI
    # Check if OpenAI API key is available
    if not os.getenv("OPENAI_API_KEY"):
        st.error("⚠️ OpenAI API key not found. Please add your API key to the .env file.")
        st.info("Example format: OPENAI_API_KEY=your_key_here")
        st.stop()
    
    # Create two columns: main chat area and sidebar info
    col1, col2 = st.columns([3, 1])
    
    with col2:
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
    
    with col1:
        # Display chat messages
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Chat input
        if prompt := st.chat_input("Ask me about financial data or Sri Lankan business insights..."):
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Display assistant response
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

# Footer
st.markdown("---")
st.caption("Multi-Company Financial Dashboard powered by Streamlit • Data refreshed quarterly")