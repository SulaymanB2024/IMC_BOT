"""Main script for running the trading data analysis pipeline."""
import argparse
import logging
from pathlib import Path

from src.utils.logging_config import setup_logging
from src.data_analysis_pipeline.config import load_config
from src.data_analysis_pipeline.data_ingestion import load_trading_data, merge_data
from src.data_analysis_pipeline.data_cleaning import validate_and_clean_data
from src.data_analysis_pipeline.feature_engineering import generate_features
from src.data_analysis_pipeline.anomaly_detection import detect_anomalies
from src.data_analysis_pipeline.statistical_analysis import run_statistical_tests
from src.data_analysis_pipeline.output_generation import save_outputs
from src.data_analysis_pipeline.insight_generation import generate_insights
from src.data_analysis_pipeline.llm_output import generate_llm_output

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Run the trading data analysis pipeline'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to configuration file'
    )
    return parser.parse_args()

def run_pipeline(config_path: str):
    """Run the full data analysis pipeline.
    
    Args:
        config_path: Path to configuration file
    """
    logger = logging.getLogger(__name__)
    logger.info("Starting trading data analysis pipeline")
    
    # Load and process data
    prices_df, orders_df = load_trading_data()
    logger.info(f"Loaded raw price data with shape: {prices_df.shape}")
    logger.info(f"Loaded raw order data with shape: {orders_df.shape}")
    
    # Merge data
    raw_data = merge_data(prices_df, orders_df)
    logger.info(f"Merged data shape: {raw_data.shape}")
    
    # Clean data
    cleaned_data = validate_and_clean_data(raw_data)
    logger.info(f"Cleaned data shape: {cleaned_data.shape}")
    
    # Generate features
    feature_data = generate_features(cleaned_data)
    logger.info(f"Generated features. Final shape: {feature_data.shape}")
    
    # Detect anomalies
    feature_data = detect_anomalies(feature_data)
    logger.info("Anomaly detection complete")
    
    # Run statistical tests
    run_statistical_tests(feature_data)
    logger.info("Statistical analysis complete")
    
    # Generate market insights
    generate_insights(feature_data)
    logger.info("Market insights generated")
    
    # Generate LLM-optimized output
    generate_llm_output(feature_data)
    logger.info("LLM-optimized output generated")
    
    # Save outputs
    save_outputs(feature_data)
    logger.info("Pipeline completed successfully")

def main():
    """Main entry point for the pipeline."""
    try:
        # Setup logging first
        setup_logging()
        logger = logging.getLogger(__name__)
        
        # Parse arguments and load config
        args = parse_args()
        config = load_config(args.config)
        
        # Run the pipeline
        run_pipeline(args.config)
        
    except Exception as e:
        # Get logger after setup (it should exist now)
        logger = logging.getLogger(__name__)
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

if __name__ == '__main__':
    main()