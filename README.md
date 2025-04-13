# IMC_BOT Data Analysis Pipeline

A robust data analysis pipeline for generating high-quality features for the IMC trading bot (givenProgram.py).

## Overview

This pipeline processes trading data to generate a comprehensive set of technical indicators, volatility measures, and anomaly flags. The processed data is saved in a format ready to be consumed by the trading bot.

## Project Structure

```
IMC_BOT/
├── data/                 # Data storage
│   ├── raw/             # Extracted raw data
│   └── processed/       # Intermediate processed data
├── notebooks/           # Analysis notebooks
│   └── exploratory_analysis.ipynb
├── src/                 # Main source code
│   ├── data_analysis_pipeline/
│   │   ├── config.py           # Configuration handling
│   │   ├── data_ingestion.py   # Data loading
│   │   ├── data_cleaning.py    # Data validation & cleaning
│   │   ├── feature_engineering.py  # Feature generation
│   │   └── output_generation.py    # Output handling
│   └── utils/
│       └── logging_config.py    # Logging setup
├── outputs/             # Generated outputs
│   ├── web_dashboard/   # Visualizations & summaries
│   └── trading_model/   # Features for trading bot
├── config.yaml         # Pipeline configuration
├── main.py            # Pipeline entry point
├── requirements.txt   # Python dependencies
└── givenProgram.py   # Trading bot (uses pipeline output)
```

## Features Generated

The pipeline generates several categories of features:

### Technical Indicators
- Simple Moving Averages (multiple windows)
- Exponential Moving Averages
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands

### Volatility Features
- Rolling standard deviation of returns
- Volatility regimes (low/medium/high)

### Lagged Features
- Price lags
- Volume lags

### Anomaly Detection
- Z-score based price anomalies
- Basic statistical flags

## Setup & Installation

1. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Configure the pipeline in `config.yaml`:
   - Set input data paths
   - Adjust feature engineering parameters
   - Configure output locations

## Running the Pipeline

Run the complete pipeline with:
```bash
python main.py
```

Or specify a custom config file:
```bash
python main.py --config custom_config.yaml
```

## Outputs

### Primary Output
- `outputs/trading_model/features.parquet`: Main feature dataset for the trading bot
  - Includes all technical indicators and derived features
  - Optimized for efficient loading
  - Ready to be consumed by givenProgram.py

### Dashboard Outputs
- `outputs/web_dashboard/`:
  - Interactive price charts with indicators
  - Technical indicator visualizations
  - Data summaries and statistics

## Configuration

The `config.yaml` file controls all aspects of the pipeline:

```yaml
input:
  zip_file: "round-2-island-data-bottle.zip"
  spreadsheet_patterns:
    - "*_prices.csv"
    - "*_orders.csv"

processing:
  time_alignment:
    resample_freq: "1T"  # 1 minute
    fill_method: "ffill"

features:
  technical_indicators:
    sma_windows: [5, 10, 20]
    rsi_period: 14
    # ... more settings ...

  volatility:
    std_windows: [5, 10, 20]
    
  # ... more feature settings ...
```

## Error Handling & Logging

- Comprehensive error handling throughout the pipeline
- Detailed logging to track processing steps
- Validation checks for data quality
- Clear error messages for troubleshooting

## Running Tests

To run the unit tests for this pipeline:

1. Install development dependencies:
   ```bash
   pip install -r requirements-dev.txt
   ```

2. Run pytest with coverage report:
   ```bash
   pytest --cov=src/data_analysis_pipeline tests/
   ```

3. Run specific test files:
   ```bash
   pytest tests/test_data_cleaning.py
   pytest tests/test_feature_engineering.py
   ```

4. Code Style and Formatting:
   ```bash
   # Check code style
   flake8 src/ tests/
   
   # Format code
   black src/ tests/
   ```

## Contributing

1. Follow the existing code structure
2. Add comprehensive docstrings
3. Implement proper error handling
4. Update configuration as needed
5. Add tests for new features

## License

MIT License - See LICENSE file for details