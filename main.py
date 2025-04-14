#!/usr/bin/env python3
"""Main script for running the trading data analysis pipeline."""
import sys
import logging
import argparse
from pathlib import Path

from src.data_analysis_pipeline.config import get_config
from src.data_analysis_pipeline.data_ingestion import load_trading_data
from src.data_analysis_pipeline.data_cleaning import validate_and_clean_data, merge_data
from src.data_analysis_pipeline.feature_engineering import generate_features
from src.data_analysis_pipeline.anomaly_detection import detect_anomalies
from src.data_analysis_pipeline.hmm_analysis import run_hmm_analysis
from src.data_analysis_pipeline.statistical_analysis import run_statistical_tests
from src.data_analysis_pipeline.visualization import generate_visualizations
from src.utils.llm_analysis import generate_llm_summary
from src.data_analysis_pipeline.output_generation import save_outputs
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)

def run_pipeline(config_path: str = "config.yaml"):
    """Run the complete data analysis pipeline."""
    try:
        # Load configuration
        config = get_config(config_path)
        
        # Load and preprocess data
        logger.info("Loading trading data...")
        raw_data = load_trading_data(config['data_paths'])
        
        # Clean and validate data
        logger.info("Cleaning and validating data...")
        cleaned_data = validate_and_clean_data(raw_data)
        merged_data = merge_data(cleaned_data)
        
        # Generate features
        logger.info("Generating features...")
        feature_data = generate_features(merged_data)
        
        # Run anomaly detection
        if config.get('anomaly_detection', {}).get('enabled', True):
            logger.info("Running anomaly detection...")
            feature_data = detect_anomalies(feature_data)
        
        # Run HMM analysis
        if config.get('hmm_analysis', {}).get('enabled', True):
            logger.info("Running HMM analysis...")
            feature_data = run_hmm_analysis(feature_data)
        
        # Run statistical tests
        if config.get('statistical_analysis', {}).get('enabled', True):
            logger.info("Running statistical analysis...")
            run_statistical_tests(feature_data)
        
        # Generate visualizations
        if config.get('visualization', {}).get('enabled', True):
            logger.info("Generating visualizations...")
            generate_visualizations(feature_data)
        
        # Save final outputs
        logger.info("Saving outputs...")
        save_outputs(feature_data)
        
        # Generate LLM summary if enabled
        if config.get('llm_output_generation', {}).get('enabled', True):
            logger.info("Generating LLM-optimized summary...")
            generate_llm_summary(feature_data)
        
        logger.info("Pipeline completed successfully")
        return feature_data
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Run the trading data analysis pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    try:
        run_pipeline(args.config)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()