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