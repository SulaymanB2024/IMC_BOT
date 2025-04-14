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
        pipeline_mode = config.get('pipeline_mode', 'full_analysis')
        logger.info(f"Running pipeline in {pipeline_mode} mode")
        
        # Core trading features pipeline (always runs)
        logger.info("Loading trading data...")
        raw_data = load_trading_data(config['data_paths'])
        
        logger.info("Cleaning and validating data...")
        cleaned_data = validate_and_clean_data(raw_data)
        merged_data = merge_data(cleaned_data)
        
        logger.info("Generating core features...")
        feature_data = generate_features(merged_data)
        
        # Anomaly detection and HMM (essential for trading signals)
        if config.get('anomaly_detection', {}).get('enabled', True):
            logger.info("Running anomaly detection...")
            feature_data = detect_anomalies(feature_data)
        
        if config.get('hmm_analysis', {}).get('enabled', True):
            logger.info("Running HMM analysis...")
            feature_data = run_hmm_analysis(feature_data)
        
        # Additional analysis modules (only in full_analysis mode)
        if pipeline_mode == 'full_analysis':
            # Statistical analysis
            if config.get('statistical_analysis', {}).get('enabled', True):
                logger.info("Running statistical analysis...")
                run_statistical_tests(feature_data)
            
            # Full visualization suite
            if config.get('visualization', {}).get('enabled', True):
                logger.info("Generating complete visualization suite...")
                generate_visualizations(feature_data)
            
            # LLM-based insights
            if config.get('llm_output_generation', {}).get('enabled', True):
                logger.info("Generating LLM-optimized summary...")
                generate_llm_summary(feature_data)
        else:
            # Minimal visualizations for trading mode
            if config.get('visualization', {}).get('enabled', True):
                logger.info("Generating essential trading visualizations...")
                generate_visualizations(
                    feature_data,
                    trading_mode=True  # Signal to generate only core visualizations
                )
        
        # Save outputs (always runs, but adapts to mode)
        logger.info("Saving outputs...")
        save_outputs(feature_data)
        
        logger.info(f"Pipeline completed successfully in {pipeline_mode} mode")
        return feature_data
        
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}", exc_info=True)
        raise

def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(description="Run the trading data analysis pipeline")
    parser.add_argument("--config", default="config.yaml", help="Path to configuration file")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    parser.add_argument("--mode", choices=['trading', 'full_analysis'], 
                       help="Override pipeline mode from config")
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else logging.INFO
    setup_logging(log_level)
    
    try:
        # If mode specified via CLI, temporarily override config
        if args.mode:
            config = get_config(args.config)
            config['pipeline_mode'] = args.mode
            logger.info(f"Pipeline mode overridden from CLI: {args.mode}")
        
        run_pipeline(args.config)
    except Exception as e:
        logger.error(f"Pipeline execution failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()