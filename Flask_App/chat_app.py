"""Standalone financial dashboard with AI assistant."""

import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
import requests
import json
from typing import List, Dict, Any
from langchain_openai import ChatOpenAI
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain.agents import Tool, AgentExecutor, create_openai_tools_agent
from langchain.tools.base import ToolException

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Financial AI Assistant",
    page_icon="💬",
    layout="wide",
)

# Custom CSS
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

# Header
st.markdown("""
<div class="finance-header">
    <h1>💬 Financial AI Assistant</h1>
    <p>Ask questions about financial performance, get analysis and recommendations</p>
</div>
""", unsafe_allow_html=True)

#################################################
# AI AGENT IMPLEMENTATION
#################################################

class DIPDAnalyzer:
    """Agent specialized in analyzing DIPD company data"""
    def __init__(self, dipd_data):
        self.data = dipd_data
        
    def analyze_metrics(self) -> Dict[str, Any]:
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
        
    def analyze_metrics(self) -> Dict[str, Any]:
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
    
    def __init__(self, companies_data):
        """Initialize the AI agent with company data."""
        self.companies_data = companies_data
        self.company_names = {company_id: data.get('name', f"Company {company_id}") 
                             for company_id, data in companies_data.items()}
        
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
        self.tools = self._setup_tools()
        
        # Create agent
        self.agent = self._create_agent()
    
    def _setup_tools(self) -> List[Tool]:
        """Set up tools for the agent."""
        
        @tool
        def analyze_dipd() -> Dict[str, Any]:
            """Analyze DIPD company metrics and performance"""
            return self.dipd_analyzer.analyze_metrics()
            
        @tool
        def analyze_rexp() -> Dict[str, Any]:
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
        
        return [
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
    
    def _create_agent(self) -> AgentExecutor:
        """Create the LangChain agent."""
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
        
        return AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True
        )
    
    def run(self, query: str) -> str:
        """Run the agent with a user query."""
        return self.agent.invoke({"input": query})["output"]
    
    def reset_memory(self):
        """Reset the agent's memory."""
        self.memory.clear()

def initialize_agent(companies_data):
    """Initialize the financial AI agent with company data."""
    return FinancialAIAgent(companies_data)

#################################################
# UI IMPLEMENTATION
#################################################

# Function to load company data
@st.cache_data
def load_companies_data():
    """Load company data from JSON files."""
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
    
    return companies_data, company_configs

# Load company data
with st.spinner("Loading financial data..."):
    all_companies_data, company_configs = load_companies_data()

# Check if OpenAI API key is available
if not os.getenv("OPENAI_API_KEY"):
    st.error("⚠️ OpenAI API key not found. Please add your API key to the .env file.")
    st.info("Example format: OPENAI_API_KEY=your_key_here")
    st.stop()

# Check if Perplexity API key is available
if not os.getenv("PERPLEXITY_API_KEY"):
    st.warning("⚠️ Perplexity API key not found. General business queries about Sri Lanka may not work properly.")

# Initialize session state for chat history if it doesn't exist
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'agent' not in st.session_state:
    with st.spinner("Initializing AI assistant..."):
        st.session_state.agent = initialize_agent(all_companies_data)

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
st.caption("Powered by LangChain, OpenAI, and Perplexity API") 