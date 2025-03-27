import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
from datetime import datetime

def apply_custom_css():
    """Apply custom CSS styling to the app"""
    st.markdown("""
    <style>
        .main {
            padding-top: 1rem;
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 2px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: #f0f2f6;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #e6f0ff;
            color: #0366d6;
            font-weight: bold;
        }
        div[data-testid="stMetricValue"] {
            font-size: 28px;
        }
        div[data-testid="stMetricDelta"] {
            font-size: 14px;
        }
        .positive {
            color: green;
        }
        .negative {
            color: red;
        }
        .company-selector {
            padding: 10px;
            background-color: #f8f9fa;
            border-radius: 5px;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_all_companies_data(company_configs):
    """Load data for all companies"""
    companies_data = {}
    
    for company_id, config in company_configs.items():
        try:
            df = pd.read_json(config["file_path"])
            df["date"] = pd.to_datetime(df["date"])
            df["year"] = df["date"].dt.year
            df["quarter"] = df["date"].dt.quarter
            df["quarter_year"] = df["date"].dt.strftime('%Y-Q') + df["quarter"].astype(str)
            df["formatted_date"] = df["date"].dt.strftime('%b %Y')
            
            # Calculate financial metrics
            df["gross_margin"] = (df["gross_profit"] / df["revenue"] * 100).round(2)
            df["operating_margin"] = (df["operating_income"] / df["revenue"] * 100).round(2)
            df["net_margin"] = (df["net_income"] / df["revenue"] * 100).round(2)
            df["efficiency_ratio"] = (df["operating_expenses"] / df["revenue"] * 100).round(2)
            
            companies_data[company_id] = df
        except Exception as e:
            st.error(f"Error loading data for {company_id}: {str(e)}")
            companies_data[company_id] = None
    
    return companies_data

def format_metric(label, value, previous_value, is_percentage=False):
    """Format KPI metrics with appropriate delta values"""
    if previous_value is not None:
        if is_percentage:
            delta = value - previous_value
            delta_text = f"{delta:.2f} pts"
        else:
            delta_percentage = ((value - previous_value) / previous_value * 100)
            delta_text = f"{delta_percentage:.1f}%"
    else:
        delta_text = None
        
    if is_percentage:
        formatted_value = f"{value:.2f}%"
    else:
        formatted_value = f"LKR {value:,.0f}"
        
    return label, formatted_value, delta_text

def display_company_metrics(year_filtered_df):
    """Display KPI metrics for the most recent quarter"""
    st.header("Key Performance Indicators")
    
    latest_data = year_filtered_df.iloc[-1]
    prev_quarter = year_filtered_df.iloc[-2] if len(year_filtered_df) > 1 else None
    
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        label, value, delta = format_metric("Revenue", latest_data["revenue"], prev_quarter["revenue"] if prev_quarter is not None else None)
        st.metric(label=label, value=value, delta=delta)

    with col2:
        label, value, delta = format_metric("Net Income", latest_data["net_income"], prev_quarter["net_income"] if prev_quarter is not None else None)
        st.metric(label=label, value=value, delta=delta)

    with col3:
        label, value, delta = format_metric("Gross Margin", latest_data["gross_margin"], prev_quarter["gross_margin"] if prev_quarter is not None else None, is_percentage=True)
        st.metric(label=label, value=value, delta=delta)

    with col4:
        label, value, delta = format_metric("Net Margin", latest_data["net_margin"], prev_quarter["net_margin"] if prev_quarter is not None else None, is_percentage=True)
        st.metric(label=label, value=value, delta=delta)

def render_overview_tab(year_filtered_df, selected_company_id, selected_company_name, enable_comparison, 
                        comparison_df, comparison_company_id, comparison_company_name, company_configs):
    """Render the overview tab content"""
    st.subheader("Financial Overview")
    
    # Revenue & Net Income chart
    fig1 = make_subplots(specs=[[{"secondary_y": True}]])
    
    # Add revenue bars
    fig1.add_trace(
        go.Bar(
            x=year_filtered_df["quarter_year"],
            y=year_filtered_df["revenue"],
            name=f"{selected_company_name} Revenue",
            marker_color=company_configs[selected_company_id]["color"]
        ),
        secondary_y=False,
    )
    
    # Add comparison company if selected
    if enable_comparison:
        comp_filtered_df = comparison_df[comparison_df["year"].isin(year_filtered_df["year"].unique())]
        
        fig1.add_trace(
            go.Bar(
                x=comp_filtered_df["quarter_year"],
                y=comp_filtered_df["revenue"],
                name=f"{comparison_company_name} Revenue",
                marker_color=company_configs[comparison_company_id]["color"],
                opacity=0.7
            ),
            secondary_y=False,
        )
    
    # Add net income line
    fig1.add_trace(
        go.Scatter(
            x=year_filtered_df["quarter_year"],
            y=year_filtered_df["net_income"],
            name=f"{selected_company_name} Net Income",
            mode='lines+markers',
            line=dict(color='rgb(220, 57, 18)', width=3)
        ),
        secondary_y=True,
    )
    
    # Add comparison company net income if selected
    if enable_comparison:
        fig1.add_trace(
            go.Scatter(
                x=comp_filtered_df["quarter_year"],
                y=comp_filtered_df["net_income"],
                name=f"{comparison_company_name} Net Income",
                mode='lines+markers',
                line=dict(color='rgb(51, 102, 204)', width=3, dash='dash')
            ),
            secondary_y=True,
        )
    
    # Set layout
    fig1.update_layout(
        title="Revenue and Net Income Trends",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        height=400,
        template="plotly_white"
    )
    
    # Set y-axis titles
    fig1.update_yaxes(title_text="Revenue (LKR)", secondary_y=False)
    fig1.update_yaxes(title_text="Net Income (LKR)", secondary_y=True)
    
    st.plotly_chart(fig1, use_container_width=True)
    
    # Expense breakdown
    st.subheader("Expense Breakdown")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Stacked area chart for expenses
        fig2 = go.Figure()
        
        fig2.add_trace(go.Scatter(
            x=year_filtered_df["quarter_year"], 
            y=year_filtered_df["COGS"],
            mode='lines',
            line=dict(width=0.5, color='rgb(131, 90, 241)'),
            stackgroup='one',
            name='COGS'
        ))
        
        fig2.add_trace(go.Scatter(
            x=year_filtered_df["quarter_year"], 
            y=year_filtered_df["operating_expenses"],
            mode='lines',
            line=dict(width=0.5, color='rgb(255, 127, 14)'),
            stackgroup='one',
            name='Operating Expenses'
        ))
        
        fig2.update_layout(
            title="Expense Composition Over Time",
            hovermode="x unified",
            height=350,
            template="plotly_white"
        )
        
        st.plotly_chart(fig2, use_container_width=True)
    
    with col2:
        # Pie chart for latest quarter expenses
        latest_data = year_filtered_df.iloc[-1]
        labels = ['COGS', 'Operating Expenses']
        values = [latest_data["COGS"], latest_data["operating_expenses"]]
        
        fig3 = go.Figure(data=[go.Pie(
            labels=labels,
            values=values,
            hole=.4,
            marker=dict(colors=['rgb(131, 90, 241)', 'rgb(255, 127, 14)'])
        )])
        
        fig3.update_layout(
            title=f"Current Quarter Expense Distribution",
            height=350
        )
        
        fig3.add_annotation(
            text=f"LKR {sum(values):,.0f}",
            x=0.5, y=0.5,
            font_size=14,
            showarrow=False
        )
        
        st.plotly_chart(fig3, use_container_width=True)

def render_revenue_analysis_tab(year_filtered_df, selected_company_id, selected_company_name, 
                               selected_years, company_configs):
    """Render the revenue analysis tab content"""
    st.subheader("Revenue Analysis")
    
    # Revenue by year
    yearly_revenue = year_filtered_df.groupby('year')['revenue'].sum().reset_index()
    
    fig4 = px.bar(
        yearly_revenue, 
        x='year', 
        y='revenue',
        title="Annual Revenue",
        text_auto='.2s',
        height=400,
        color_discrete_sequence=[company_configs[selected_company_id]["color"]]
    )
    
    fig4.update_traces(textposition='outside')
    fig4.update_layout(uniformtext_minsize=8, uniformtext_mode='hide')
    
    st.plotly_chart(fig4, use_container_width=True)
    
    # Quarter-by-quarter YoY revenue growth
    if len(selected_years) > 1:
        st.subheader("Quarter-by-Quarter Year-over-Year Comparison")
        
        # Prepare data for each quarter
        quarters_data = []
        for quarter in [1, 2, 3, 4]:
            quarter_df = year_filtered_df[year_filtered_df["quarter"] == quarter]
            quarters_data.append(quarter_df)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Q1 and Q2
            for i in range(2):
                q_df = quarters_data[i]
                if not q_df.empty:
                    fig = px.bar(
                        q_df, 
                        x='year', 
                        y='revenue',
                        title=f"Q{i+1} Revenue Comparison",
                        text_auto='.2s',
                        height=300,
                        color_discrete_sequence=[company_configs[selected_company_id]["color"]]
                    )
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)
                
        with col2:
            # Q3 and Q4
            for i in range(2, 4):
                q_df = quarters_data[i]
                if not q_df.empty:
                    fig = px.bar(
                        q_df, 
                        y='year', 
                        x='revenue',
                        orientation='h',
                        title=f"Q{i+1} Revenue Comparison",
                        text_auto='.2s',
                        height=300,
                        color_discrete_sequence=[company_configs[selected_company_id]["color"]]
                    )
                    fig.update_traces(textposition='outside')
                    st.plotly_chart(fig, use_container_width=True)

def render_margin_analysis_tab(year_filtered_df, selected_company_name, enable_comparison, 
                              comparison_df, comparison_company_name, selected_years):
    """Render the margin analysis tab content"""
    st.subheader("Margin Analysis")
    
    # Create a line chart for margins
    fig5 = go.Figure()
    
    fig5.add_trace(go.Scatter(
        x=year_filtered_df["quarter_year"],
        y=year_filtered_df["gross_margin"],
        mode='lines+markers',
        name='Gross Margin',
        line=dict(color='rgb(44, 160, 44)', width=2)
    ))
    
    fig5.add_trace(go.Scatter(
        x=year_filtered_df["quarter_year"],
        y=year_filtered_df["operating_margin"],
        mode='lines+markers',
        name='Operating Margin',
        line=dict(color='rgb(214, 39, 40)', width=2)
    ))
    
    fig5.add_trace(go.Scatter(
        x=year_filtered_df["quarter_year"],
        y=year_filtered_df["net_margin"],
        mode='lines+markers',
        name='Net Margin',
        line=dict(color='rgb(148, 103, 189)', width=2)
    ))
    
    # Add comparison company if selected
    if enable_comparison:
        comp_filtered_df = comparison_df[comparison_df["year"].isin(selected_years)]
        
        fig5.add_trace(go.Scatter(
            x=comp_filtered_df["quarter_year"],
            y=comp_filtered_df["gross_margin"],
            mode='lines+markers',
            name=f'{comparison_company_name} Gross Margin',
            line=dict(color='rgb(44, 160, 44)', width=2, dash='dash')
        ))
        
        fig5.add_trace(go.Scatter(
            x=comp_filtered_df["quarter_year"],
            y=comp_filtered_df["net_margin"],
            mode='lines+markers',
            name=f'{comparison_company_name} Net Margin',
            line=dict(color='rgb(148, 103, 189)', width=2, dash='dash')
        ))
    
    fig5.update_layout(
        title="Margin Trends",
        xaxis_title="Quarter",
        yaxis_title="Percentage (%)",
        hovermode="x unified",
        height=400,
        template="plotly_white"
    )
    
    st.plotly_chart(fig5, use_container_width=True)
    
    # Scatter plot relating revenue to net income
    fig6 = px.scatter(
        year_filtered_df, 
        x="revenue", 
        y="net_income",
        size="operating_income",
        color="year",
        hover_name="quarter_year",
        text="quarter",
        height=500,
        title="Revenue vs. Net Income Relationship",
        trendline="ols"
    )
    
    fig6.update_traces(textposition='top center')
    
    st.plotly_chart(fig6, use_container_width=True)
    
    # Margin efficiency analysis
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Margin to Expense Ratio")
        
        fig7 = px.scatter(
            year_filtered_df,
            x="efficiency_ratio",
            y="net_margin",
            size="revenue",
            color="year",
            hover_name="quarter_year",
            height=400,
            title="Expense Efficiency vs Net Margin"
        )
        
        st.plotly_chart(fig7, use_container_width=True)
    
    with col2:
        st.subheader("Gross to Net Margin")
        
        fig8 = px.scatter(
            year_filtered_df,
            x="gross_margin",
            y="net_margin",
            size="revenue",
            color="year",
            hover_name="quarter_year",
            height=400,
            title="Gross Margin vs Net Margin",
            trendline="ols"
        )
        
        st.plotly_chart(fig8, use_container_width=True)

def render_quarterly_details_tab(year_filtered_df):
    """Render the quarterly details tab content"""
    st.header("Quarter-Specific Analysis")
    
    # Quarter selector
    col1, col2 = st.columns([1, 3])
    
    with col1:
        selected_year = st.selectbox("Year", options=sorted(year_filtered_df["year"].unique(), reverse=True))
    
    with col2:
        selected_quarter = st.radio("Quarter", options=[1, 2, 3, 4], horizontal=True)
    
    # Get the data for the selected quarter
    quarter_mask = (year_filtered_df["year"] == selected_year) & (year_filtered_df["quarter"] == selected_quarter)
    if not year_filtered_df[quarter_mask].empty:
        quarter_data = year_filtered_df[quarter_mask].iloc[0]
        
        # Find previous quarter
        if selected_quarter > 1:
            prev_quarter_mask = (year_filtered_df["year"] == selected_year) & (year_filtered_df["quarter"] == selected_quarter - 1)
        else:
            prev_quarter_mask = (year_filtered_df["year"] == selected_year - 1) & (year_filtered_df["quarter"] == 4)
        
        has_prev_quarter = not year_filtered_df[prev_quarter_mask].empty
        if has_prev_quarter:
            prev_quarter = year_filtered_df[prev_quarter_mask].iloc[0]
        else:
            prev_quarter = None
        
        # Find year-over-year quarter
        yoy_quarter_mask = (year_filtered_df["year"] == selected_year - 1) & (year_filtered_df["quarter"] == selected_quarter)
        has_yoy_quarter = not year_filtered_df[yoy_quarter_mask].empty
        if has_yoy_quarter:
            yoy_quarter = year_filtered_df[yoy_quarter_mask].iloc[0]
        else:
            yoy_quarter = None
        
        # Display quarterly metrics
        st.subheader(f"Financial Metrics for {selected_year} Q{selected_quarter}")
        
        # Prepare metrics
        metrics = [
            {"name": "Revenue", "key": "revenue", "format": "LKR "},
            {"name": "Gross Profit", "key": "gross_profit", "format": "LKR "},
            {"name": "Operating Income", "key": "operating_income", "format": "LKR "},
            {"name": "Net Income", "key": "net_income", "format": "LKR "},
            {"name": "Gross Margin", "key": "gross_margin", "format": "%"},
            {"name": "Operating Margin", "key": "operating_margin", "format": "%"},
            {"name": "Net Margin", "key": "net_margin", "format": "%"},
        ]
        
        # Create data table
        data_rows = []
        
        for metric in metrics:
            row = {
                "Metric": metric["name"],
                "Value": f"{metric['format']}{quarter_data[metric['key']]:,.2f}" if metric["format"] == "LKR " else f"{quarter_data[metric['key']]:.2f}%"
            }
            
            # Calculate QoQ and YoY changes
            if has_prev_quarter:
                if metric["format"] == "LKR ":
                    qoq_change = ((quarter_data[metric["key"]] - prev_quarter[metric["key"]]) / prev_quarter[metric["key"]] * 100).round(1)
                    row["QoQ Change"] = f"{qoq_change:.1f}%"
                else:
                    qoq_change = (quarter_data[metric["key"]] - prev_quarter[metric["key"]]).round(2)
                    row["QoQ Change"] = f"{qoq_change:.2f} pts"
            else:
                row["QoQ Change"] = "N/A"
            
            if has_yoy_quarter:
                if metric["format"] == "LKR ":
                    yoy_change = ((quarter_data[metric["key"]] - yoy_quarter[metric["key"]]) / yoy_quarter[metric["key"]] * 100).round(1)
                    row["YoY Change"] = f"{yoy_change:.1f}%"
                else:
                    yoy_change = (quarter_data[metric["key"]] - yoy_quarter[metric["key"]]).round(2)
                    row["YoY Change"] = f"{yoy_change:.2f} pts"
            else:
                row["YoY Change"] = "N/A"
            
            data_rows.append(row)
        
        # Convert to DataFrame for display
        metrics_df = pd.DataFrame(data_rows)
        
        # Show the table
        st.dataframe(metrics_df, hide_index=True, width=800)
        
        # Expenses breakdown pie chart
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Performance radar chart
            categories = ['Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'Gross Margin']
            
            fig = go.Figure()
            
            # Current quarter
            fig.add_trace(go.Scatterpolar(
                r=[
                    quarter_data["revenue"] / 1e6,
                    quarter_data["gross_profit"] / 1e6,
                    quarter_data["operating_income"] / 1e6,
                    quarter_data["net_income"] / 1e6,
                    quarter_data["gross_margin"],
                ],
                theta=categories,
                fill='toself',
                name=f'{selected_year} Q{selected_quarter}'
            ))
            
            if has_yoy_quarter:
                fig.add_trace(go.Scatterpolar(
                    r=[
                        yoy_quarter["revenue"] / 1e6,
                        yoy_quarter["gross_profit"] / 1e6,
                        yoy_quarter["operating_income"] / 1e6,
                        yoy_quarter["net_income"] / 1e6,
                        yoy_quarter["gross_margin"],
                    ],
                    theta=categories,
                    fill='toself',
                    name=f'{selected_year-1} Q{selected_quarter}'
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                    )
                ),
                title="Quarterly Performance Comparison (LKR in millions, except margins in %)",
                showlegend=True,
                height=450
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Pie chart for expenses
            labels = ['COGS', 'Operating Expenses']
            values = [quarter_data["COGS"], quarter_data["operating_expenses"]]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels,
                values=values,
                hole=.4
            )])
            
            fig.update_layout(
                title="Expense Breakdown",
                height=450
            )
            
            total_expenses = sum(values)
            fig.add_annotation(
                text=f"LKR {total_expenses:,.0f}",
                x=0.5, y=0.5,
                font_size=14,
                showarrow=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning(f"No data available for {selected_year} Q{selected_quarter}")

def render_company_comparison(year_filtered_df, comparison_df, selected_company_name, 
                              comparison_company_name, selected_company_id, comparison_company_id, 
                              company_configs, selected_years):
    """Render the company comparison section"""
    st.header("Company Comparison Analysis")
    
    # Get the comparison data
    comp_year_filtered_df = comparison_df[comparison_df["year"].isin(selected_years)]
    
    # Create metrics for comparison
    metrics_to_compare = ["revenue", "net_income", "gross_margin", "net_margin"]
    metric_names = {
        "revenue": "Revenue",
        "net_income": "Net Income", 
        "gross_margin": "Gross Margin", 
        "net_margin": "Net Margin"
    }
    
    # Show comparative charts
    for i, metric in enumerate(metrics_to_compare):
        if i % 2 == 0:
            col1, col2 = st.columns(2)
        
        fig = go.Figure()
        
        # Primary company
        fig.add_trace(go.Scatter(
            x=year_filtered_df["quarter_year"],
            y=year_filtered_df[metric],
            mode='lines+markers',
            name=selected_company_name,
            line=dict(color=company_configs[selected_company_id]["color"], width=2)
        ))
        
        # Comparison company
        fig.add_trace(go.Scatter(
            x=comp_year_filtered_df["quarter_year"],
            y=comp_year_filtered_df[metric],
            mode='lines+markers',
            name=comparison_company_name,
            line=dict(color=company_configs[comparison_company_id]["color"], width=2)
        ))
        
        fig.update_layout(
            title=f"{metric_names[metric]} Comparison",
            xaxis_title="Quarter",
            yaxis_title="Value",
            hovermode="x unified",
            height=350,
            template="plotly_white"
        )
        
        if metric in ["gross_margin", "net_margin"]:
            fig.update_yaxes(title_text="Percentage (%)")
        else:
            fig.update_yaxes(title_text="Value (LKR)")
            
        if i % 2 == 0:
            col1.plotly_chart(fig, use_container_width=True)
        else:
            col2.plotly_chart(fig, use_container_width=True)

def apply_chat_css():
    """Apply custom CSS styling for the chat interface"""
    st.markdown("""
    <style>
        .chat-message {
            padding: 1.5rem; 
            border-radius: 0.5rem; 
            margin-bottom: 1rem; 
            display: flex;
            flex-direction: column;
        }
        .chat-message.user {
            background-color: #f0f2f6;
        }
        .chat-message.assistant {
            background-color: #e6f0ff;
        }
        .chat-message .avatar {
            width: 40px;
            height: 40px;
            margin-right: 10px;
        }
        .chat-message .content {
            flex-grow: 1;
        }
        .finance-header {
            background-color: #0366d6;
            color: white;
            padding: 1.5rem;
            border-radius: 0.5rem;
            margin-bottom: 2rem;
        }
        .example-questions {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            margin-bottom: 1rem;
        }
    </style>
    """, unsafe_allow_html=True)

class DIPDAnalyzer:
    """Agent specialized in analyzing DIPD company data"""
    def __init__(self, dipd_data):
        self.data = dipd_data
        
    def analyze_metrics(self) -> dict:
        """Analyze latest DIPD metrics"""
        latest = self.data.iloc[-1]
        return {
            "revenue": latest["revenue"],
            "net_income": latest["net_income"],
            "margins": {
                "gross": latest["gross_margin"],
                "operating": latest["operating_margin"],
                "net": latest["net_margin"]
            }
        }

class REXPAnalyzer:
    """Agent specialized in analyzing REXP company data"""
    def __init__(self, rexp_data):
        self.data = rexp_data
        
    def analyze_metrics(self) -> dict:
        """Analyze latest REXP metrics"""
        latest = self.data.iloc[-1]
        return {
            "revenue": latest["revenue"], 
            "net_income": latest["net_income"],
            "margins": {
                "gross": latest["gross_margin"],
                "operating": latest["operating_margin"], 
                "net": latest["net_margin"]
            }
        }

class FinancialAIAgent:
    """AI Agent for analyzing financial data and answering questions."""
    
    def __init__(self, companies_data, company_configs=None):
        """Initialize the AI agent with company data."""
        import os
        from langchain_openai import ChatOpenAI
        from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
        from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
        from langchain.memory import ConversationBufferMemory
        from langchain.tools import tool
        from langchain.tools.base import ToolException
        import requests
        
        self.companies_data = companies_data
        
        # Handle company_configs if it's not provided
        if company_configs is None:
            # Create a default mapping using just the company IDs
            self.company_names = {company_id: f"Company {company_id}" 
                                for company_id in companies_data.keys()}
        else:
            # Use the provided company_configs
            self.company_names = {company_id: config.get('name', f"Company {company_id}") 
                                for company_id, config in company_configs.items()}
        
        # Initialize specialized analyzers
        self.dipd_analyzer = DIPDAnalyzer(companies_data["DIPD"])
        self.rexp_analyzer = REXPAnalyzer(companies_data["REXP"])
        
        # Initialize OpenAI LLM
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0.2,
        )
        
        # Initialize memory
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True
        )
        
        # Set up tools
        @tool
        def analyze_dipd() -> dict:
            """Analyze DIPD company metrics and performance"""
            return self.dipd_analyzer.analyze_metrics()
            
        @tool
        def analyze_rexp() -> dict:
            """Analyze REXP company metrics and performance"""
            return self.rexp_analyzer.analyze_metrics()
            
        @tool
        def get_sri_lanka_business_info(query: str) -> str:
            """Get information about businesses in Sri Lanka using Perplexity API"""
            try:
                api_key = os.getenv("PERPLEXITY_API_KEY")
                if not api_key:
                    raise ToolException("Perplexity API key not found")
                
                headers = {
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                }
                
                data = {
                    "model": "r1-1776",
                    "messages": [
                        {
                            "role": "user", 
                            "content": f"Information about {query} in Sri Lanka business context. Provide specific, factual information."
                        }
                    ],
                    "max_tokens": 300
                }
                
                response = requests.post(
                    "https://api.perplexity.ai/chat/completions",
                    headers=headers,
                    json=data
                )
                
                if response.status_code == 200:
                    result = response.json()
                    content = result.get("choices", [{}])[0].get("message", {}).get("content", "No information found")
                    return content
                else:
                    raise ToolException(f"API Error: {response.status_code}")
                    
            except Exception as e:
                raise ToolException(f"Error querying Perplexity API: {str(e)}")
        
        self.tools = [
            Tool(
                name="analyze_dipd",
                func=analyze_dipd,
                description="Get detailed analysis of DIPD company performance"
            ),
            Tool(
                name="analyze_rexp", 
                func=analyze_rexp,
                description="Get detailed analysis of REXP company performance"
            ),
            Tool(
                name="get_sri_lanka_business_info",
                func=get_sri_lanka_business_info,
                description="Search for business information about Sri Lanka"
            )
        ]
        
        # Create agent
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a Financial Analysis Assistant specializing in Sri Lankan companies.
            
            You have access to:
            1. DIPD company analyzer
            2. REXP company analyzer  
            3. Sri Lanka business information search
            
            Use these tools to provide detailed financial analysis and insights.
            Always support your analysis with specific data and metrics.
            """),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        agent = create_openai_tools_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )
    
    def run(self, query: str) -> str:
        """Run the agent with a user query."""
        return self.agent_executor.invoke({"input": query})["output"]
    
    def reset_memory(self):
        """Reset the agent's memory."""
        self.memory.clear()

def initialize_agent(companies_data, company_configs=None):
    """Initialize the financial AI agent with company data."""
    return FinancialAIAgent(companies_data, company_configs) 