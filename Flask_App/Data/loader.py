"""Data loading functions for the financial dashboard."""

import pandas as pd
import streamlit as st
from datetime import datetime

def load_all_companies_data(company_configs):
    """Load and process data for all companies."""
    @st.cache_data
    def _load_data():
        companies_data = {}
        
        for company_id, config in company_configs.items():
            try:
                df = pd.read_json(config["file_path"])
                df = process_company_data(df)
                companies_data[company_id] = df
            except Exception as e:
                st.error(f"Error loading data for {company_id}: {str(e)}")
                companies_data[company_id] = None
        
        return companies_data
    
    return _load_data()

def process_company_data(df):
    """Process raw company data to add derived fields."""
    # Convert date fields
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
    
    return df

def filter_by_date_range(df, start_date, end_date):
    """Filter dataframe by date range."""
    return df[(df["date"] >= pd.Timestamp(start_date)) & 
              (df["date"] <= pd.Timestamp(end_date))]

def filter_by_years(df, years):
    """Filter dataframe by selected years."""
    return df[df["year"].isin(years)] 