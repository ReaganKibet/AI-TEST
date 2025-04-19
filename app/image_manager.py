import os
import uuid
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
from PIL import Image
import io
import base64

class ImageManager:
    """
    Handles image generation, storage, and metadata management.
    """
    def __init__(self, output_dir="outputs"):
        """
        Initialize the ImageManager with a specified output directory.
        
        Args:
            output_dir (str): Directory to store generated images and metadata
        """
        self.output_dir = output_dir
        self.images_dir = os.path.join(output_dir, "images")
        self.thumbnails_dir = os.path.join(output_dir, "thumbnails")
        self.metadata_dir = os.path.join(output_dir, "metadata")
        
        # Create necessary directories
        os.makedirs(self.images_dir, exist_ok=True)
        os.makedirs(self.thumbnails_dir, exist_ok=True)
        os.makedirs(self.metadata_dir, exist_ok=True)
        
    def save_image_from_base64(self, base64_data: str, prompt: str) -> Dict[str, Any]:
        """
        Save an image from base64 data, generate thumbnail, and store metadata.
        
        Args:
            base64_data (str): Base64 encoded image data
            prompt (str): The prompt used to generate the image
            
        Returns:
            Dict[str, Any]: Metadata about the saved image
        """
        try:
            # Generate a unique ID for this image
            image_id = str(uuid.uuid4())
            timestamp = datetime.now().isoformat()
            
            # Decode base64 data
            if "base64," in base64_data:
                base64_data = base64_data.split("base64,")[1]
            
            image_data = base64.b64decode(base64_data)
            image = Image.open(io.BytesIO(image_data))
            
            # Save full-size image
            image_filename = f"{image_id}.png"
            image_path = os.path.join(self.images_dir, image_filename)
            image.save(image_path, format="PNG", optimize=True)
            
            # Generate and save thumbnail
            thumbnail_size = (256, 256)
            thumbnail = image.copy()
            thumbnail.thumbnail(thumbnail_size)
            thumbnail_path = os.path.join(self.thumbnails_dir, image_filename)
            thumbnail.save(thumbnail_path, format="PNG", optimize=True)
            
            # Create metadata
            metadata = {
                "id": image_id,
                "timestamp": timestamp,
                "prompt": prompt,
                "image_path": image_path,
                "thumbnail_path": thumbnail_path,
                "width": image.width,
                "height": image.height
            }
            
            # Save metadata
            metadata_path = os.path.join(self.metadata_dir, f"{image_id}.json")
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2)
                
            logging.info(f"Image saved with ID: {image_id}")
            return metadata
            
        except Exception as e:
            logging.error(f"Error saving image: {e}")
            raise
    
    def image_to_base64(self, image_path: str) -> str:
        """
        Convert an image file to base64 encoding.
        
        Args:
            image_path (str): Path to the image file
            
        Returns:
            str: Base64 encoded image data
        """
        try:
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
            return encoded_string
        except Exception as e:
            logging.error(f"Error converting image to base64: {e}")
            raise