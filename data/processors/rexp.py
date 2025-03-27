import pandas as pd
import pdfplumber
import json
import re
import os
from pathlib import Path
from langchain.chat_models import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
from dotenv import load_dotenv

class FinancialDataProcessor:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        
        # Define paths using relative paths from project root
        self.project_root = Path(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        self.input_folder = self.project_root / "data" / "raw" / "rexp"
        self.output_folder = self.project_root / "data" / "interim" / "rexp_final"
        self.csv_output_path = self.output_folder / "combined_financial_data.csv"
        
        # Create output directory if it doesn't exist
        self.output_folder.mkdir(parents=True, exist_ok=True)
        
        # Initialize storage for processed data
        self.all_data = []

    def extract_specific_page(self, pdf_path: str, page_number: int = 3) -> tuple:
        """Extract tables and text from a specific PDF page."""
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_number]
            tables = page.extract_tables()
            text = page.extract_text()
            return tables, text

    def clean_table_with_langchain(self, page_data: str) -> str:
        """Process page data using LangChain and GPT model."""
        try:
            chat = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                openai_api_key=os.getenv("OPENAI_API_KEY")
            )

            messages = [
                SystemMessage(content="""You are a data extraction expert. Focus on:
                - Extract the financial data from the table
                - Return the data in JSON format with specific metric names as keys
                - Include period_end_date, year, and month information
                - Ensure values are numeric integers"""),
                HumanMessage(content=self._create_extraction_prompt(page_data))
            ]

            response = chat(messages)
            return response.content

        except Exception as e:
            print(f"Data extraction failed: {str(e)}")
            return None

    def _create_extraction_prompt(self, page_data: str) -> str:
        """Create the extraction prompt for the LLM."""
        return f"""Please analyze this financial report and extract:
        1. The financial metrics from the table
        2. Include the reporting period information
        3. Return the data in JSON format with the following structure:
        
        {{
            "period_end_date": "YYYY-MM-DD",
            "year": YYYY,
            "month": MM,
            "Revenue": value,
            "Cost_of_Sales": value,
            "Gross_Profit": value,
            "Distribution_Costs": value,
            "Administrative_Expenses": value,
            "Profit_from_Operations": value,
            "Profit_for_the_Period": value
        }}
        
        Table data:
        {page_data}"""

    def process_single_pdf(self, pdf_file: Path) -> None:
        """Process a single PDF file and extract financial data."""
        output_file = self.output_folder / f"{pdf_file.stem}.json"
        print(f"\nProcessing: {pdf_file.name}")

        tables, text = self.extract_specific_page(str(pdf_file))
        json_data = self.clean_table_with_langchain(text)

        if json_data:
            self._handle_json_data(json_data, pdf_file, output_file)
        else:
            print(f"Failed to extract data from {pdf_file.name}")

    def _handle_json_data(self, json_data: str, pdf_file: Path, output_file: Path) -> None:
        """Handle the extracted JSON data."""
        match = re.search(r'```json\n(.*?)\n```', json_data, re.DOTALL)
        if match:
            cleaned_json = match.group(1).strip()
            try:
                extracted_clean = json.loads(cleaned_json)
                
                # Save to JSON file
                with open(output_file, 'w') as f:
                    json.dump(extracted_clean, f, indent=4)
                
                self.all_data.append(extracted_clean)
                print(f"Data saved to: {output_file}")
                
            except json.JSONDecodeError as e:
                print(f"JSON parsing error for {pdf_file.name}: {str(e)}")
        else:
            print(f"No valid JSON found in {pdf_file.name}")

    def save_to_csv(self) -> None:
        """Save all processed data to a CSV file."""
        if self.all_data:
            df = pd.DataFrame(self.all_data)
            
            if 'period_end_date' in df.columns:
                df['period_end_date'] = pd.to_datetime(df['period_end_date'])
                df = df.sort_values('period_end_date', ascending=False)
            
            df.to_csv(self.csv_output_path, index=False)
            print(f"\nData saved to CSV: {self.csv_output_path}")
        else:
            print("\nNo data was found to save")

    def process_all_files(self) -> None:
        """Process all PDF files in the input directory."""
        for pdf_file in self.input_folder.glob("*.pdf"):
            self.process_single_pdf(pdf_file)
        
        self.save_to_csv()
        print("\nProcessing complete for all files.")

def main():
    processor = FinancialDataProcessor()
    processor.process_all_files()

if __name__ == "__main__":
    main()
