import logging
import os
import sys
from datetime import datetime

def configure_logging():
    """Configure logging with timestamps, levels, and file output."""
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Generate log filename with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"logs/app_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Log startup information
    logging.info("=" * 50)
    logging.info("Application starting up")
    logging.info(f"Log file: {log_file}")
    logging.info("=" * 50)