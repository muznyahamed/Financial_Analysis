import pandas as pd
import pdfplumber
import re 
import json
import os
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Define paths
pdf_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "raw", "dipd"))
output_dir = Path(os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "interim", "dipd_fianl"))

# Create output directory if it doesn't exist
output_dir.mkdir(parents=True, exist_ok=True)

def extract_specific_page(pdf_path, page_number):
    with pdfplumber.open(pdf_path) as pdf:
        page = pdf.pages[page_number]
        text = page.extract_text()
        return text

def clean_table_with_langchain(page_data):
    try:
        chat = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            openai_api_key=os.getenv("OPENAI_API_KEY")
        )

        messages = [
            SystemMessage(content="""You are a financial data extraction expert. Extract only:
            - The unaudited 3-month financial data for the latest year
            - Return only the following parameters in JSON format:
              - 'revenue', 'cost_of_sales', 'gross_profit'
              - 'distribution_costs', 'administrative_expenses', 'other_income_and_gains'
              - 'profit_for_the_period', 'year', 'month', 'period_end_date'
            - Ensure a structured JSON output"""),
            HumanMessage(content=f"""Extract the required financial data from the following report:
            {page_data}
            
            Return the JSON object containing:
            - 'period_end_date', 'year', 'month'
            - 'revenue', 'cost_of_sales', 'gross_profit'
            - 'distribution_costs', 'administrative_expenses', 'other_income_and_gains'
            - 'profit_for_the_period'""")
        ]

        response = chat(messages)
        return response.content

    except Exception as e:
        print(f"Data extraction failed: {str(e)}")
        return None

def main():
    # Process all PDFs in the directory
    for pdf_file in pdf_dir.glob("*.pdf"):
        print(f"\nProcessing {pdf_file.name}...")
        
        # Extract data from page 2
        text = extract_specific_page(str(pdf_file), 2)
        json_data = clean_table_with_langchain(text)

        if json_data:
            # Extract JSON from response
            match = re.search(r'```json\n(.+?)\n```', json_data, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    # Parse JSON and create DataFrame
                    data_dict = json.loads(json_str)
                    df = pd.DataFrame([data_dict])
                    
                    # Create output filename with same name as PDF
                    output_file = output_dir / f"{pdf_file.stem}.csv"
                    
                    # Save DataFrame to CSV
                    df.to_csv(output_file, index=False)
                    print(f"Saved data to {output_file}")
                    
                except json.JSONDecodeError as e:
                    print(f"Error parsing JSON for {pdf_file.name}: {e}")
            else:
                print(f"No valid JSON found in response for {pdf_file.name}")
        else:
            print(f"Failed to extract data from {pdf_file.name}")

if __name__ == "__main__":
    main()
