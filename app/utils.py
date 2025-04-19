import logging
import functools
import time
from typing import Callable, Any, TypeVar, cast

T = TypeVar('T')

def retry(max_attempts: int = 3, delay_seconds: int = 2) -> Callable:
    """
    Retry decorator for functions that might fail.
    
    Args:
        max_attempts (int): Maximum number of retry attempts
        delay_seconds (int): Delay between retries in seconds
        
    Returns:
        Callable: Decorated function with retry logic
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            attempts = 0
            last_exception = None
            
            while attempts < max_attempts:
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    attempts += 1
                    last_exception = e
                    logging.warning(f"Attempt {attempts}/{max_attempts} failed: {e}")
                    if attempts < max_attempts:
                        logging.info(f"Retrying in {delay_seconds} seconds...")
                        time.sleep(delay_seconds)
            
            logging.error(f"All {max_attempts} attempts failed!")
            raise last_exception
            
        return cast(Callable[..., T], wrapper)
    return decorator


def validate_image(base64_str: str) -> bool:
    """
    Validate if a base64 string represents a valid image.
    
    Args:
        base64_str (str): Base64-encoded image data
        
    Returns:
        bool: True if valid, False otherwise
    """
    import base64
    import io
    from PIL import Image
    
    try:
        # Remove data URL prefix if present
        if "base64," in base64_str:
            base64_str = base64_str.split("base64,")[1]
            
        # Decode base64
        image_data = base64.b64decode(base64_str)
        
        # Attempt to open as image
        Image.open(io.BytesIO(image_data))
        return True
    except Exception as e:
        logging.warning(f"Invalid image data: {e}")
        return False


def load_json_config(config_path: str) -> dict:
    """
    Load a JSON configuration file.
    
    Args:
        config_path (str): Path to JSON config file
        
    Returns:
        dict: Loaded configuration
    """
    import json
    import os
    
    try:
        if not os.path.exists(config_path):
            logging.warning(f"Config file not found: {config_path}")
            return {}
            
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        logging.error(f"Error loading config file: {e}")
        return {}