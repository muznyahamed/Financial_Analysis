# Financial Data Analysis and Visualization Platform

## Project Overview

This project is a comprehensive financial data analysis platform that allows users to process, analyze, visualize, and compare financial data from multiple companies. The platform includes data extraction from financial reports, predictive modeling, and an interactive dashboard with AI-powered chat assistance.

## Features

- **Data Extraction**: Automated extraction of financial data from PDF reports
- **Data Processing**: Cleaning and structuring financial data for analysis
- **Predictive Modeling**: Machine learning models to predict future financial performance
- **Interactive Dashboard**: Visualize and compare key financial metrics
- **AI Chat Assistant**: Ask questions about financial data and get insights

## Project Structure
├── data/
│ ├── raw/ # Original PDF reports downloaded from CSE
│ │ ├── dipd/ # Dipped Products PLC reports
│ │ └── rexp/ # Richard Pieris Exports PLC reports
│ ├── interim/ # Intermediate extracted data
│ └── processed/ # Final cleaned datasets
│ ├── quarterly/ # Quarterly data points
│ └── annual/ # Aggregated annual data
├── notebooks/
│ ├── 1.0-data-exploration.ipynb
│ ├── 2.0-data-processing.ipynb
│ ├── 3.0-visualization.ipynb
│ └── 4.0-forecasting.ipynb
├── src/
│ ├── data/
│ │ ├── init.py
│ │ ├── scraper.py # PDF scraping functionality
│ │ ├── extractor.py # Data extraction logic
│ │ └── processor.py # Data cleaning and structuring
│ ├── features/
│ │ ├── init.py
│ │ └── build_features.py # Feature engineering for analysis
│ ├── visualization/
│ │ ├── init.py
│ │ └── visualize.py # Visualization components
│ └── models/
│ ├── init.py
│ ├── train_model.py # Model training
│ └── predict_model.py # Forecasting implementation
├── dashboard/
│ ├── app.py # Main dashboard application
│ ├── assets/ # CSS, images, etc.
│ └── components/ # Dashboard UI components
├── tests/
│ ├── test_scraper.py
│ ├── test_processor.py
│ └── test_models.py
├── docs/
│ ├── methodology.md # Detailed methodology documentation
│ ├── data_dictionary.md # Description of all data fields
│ └── user_guide.md # Dashboard usage instructions
├── reports/
│ ├── figures/ # Generated visualizations
│ └── final_report.md # Final analysis and findings
├── .gitignore
├── requirements.txt # Project dependencies
├── setup.py # Package installation
├── LICENSE
└── README.md # Project overview

The dashboard will be accessible at `http://localhost:8501` in your web browser.

## Using the Dashboard

1. **Select a Company**: Choose from available companies in the sidebar
2. **Apply Filters**: Filter data by date range or specific years
3. **Compare Companies**: Enable the comparison feature to analyze multiple companies
4. **Explore Tabs**: Navigate between analysis tabs for different perspectives
5. **Chat with AI**: Click "Switch to Chat Assistant" to ask questions in natural language

## Future Enhancements

- Support for additional companies
- More advanced predictive models
- Export of analysis reports
- Mobile optimization
- Integration with real-time financial data APIs

## Conclusion

This financial analysis platform demonstrates the power of combining data processing, machine learning, and interactive visualization to gain insights from financial data. The inclusion of an AI assistant makes complex financial analysis accessible to users through natural language queries.