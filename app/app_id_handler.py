# app/app_id_handler.py
import logging
import os
import requests
from typing import Dict, List, Optional

class AppIDManager:
    """
    Manages external app IDs and provides fallback mechanisms
    when they are unreachable or invalid.
    """
    
    def __init__(self):
        """Initialize the app ID manager with environment variables or defaults"""
        # Get app IDs from environment variables
        self.text_to_image_app_id = os.environ.get("TEXT_TO_IMAGE_APP_ID", "")
        self.image_to_3d_app_id = os.environ.get("IMAGE_TO_3D_APP_ID", "")
        
        # Default fallback app IDs (should be replaced with actual working IDs)
        self.default_app_ids = {
            "text_to_image": [
                "f0997a01-d6d3-a5fe-53d8-561300318557",  # Original ID
                "text2img-fallback-id-1",                 # Additional fallback IDs
                "text2img-fallback-id-2"
            ],
            "image_to_3d": [
                "69543f29-4d41-4afc-7f29-3d51591f11eb",   # Original ID
                "img23d-fallback-id-1",                   # Additional fallback IDs
                "img23d-fallback-id-2"
            ]
        }
        
        # Track which app IDs are working
        self.working_app_ids = {
            "text_to_image": None,
            "image_to_3d": None
        }
        
    def get_text_to_image_app_id(self) -> str:
        """Get a working text-to-image app ID or return empty if none works"""
        if self.working_app_ids["text_to_image"]:
            return self.working_app_ids["text_to_image"]
            
        # Try the configured ID first
        if self.text_to_image_app_id and self._check_app_id(self.text_to_image_app_id):
            self.working_app_ids["text_to_image"] = self.text_to_image_app_id
            return self.text_to_image_app_id
            
        # Try default IDs
        for app_id in self.default_app_ids["text_to_image"]:
            if self._check_app_id(app_id):
                self.working_app_ids["text_to_image"] = app_id
                return app_id
                
        # No working app ID found
        logging.warning("No working text-to-image app ID found. Using mock service.")
        return ""
        
    def get_image_to_3d_app_id(self) -> str:
        """Get a working image-to-3D app ID or return empty if none works"""
        if self.working_app_ids["image_to_3d"]:
            return self.working_app_ids["image_to_3d"]
            
        # Try the configured ID first
        if self.image_to_3d_app_id and self._check_app_id(self.image_to_3d_app_id):
            self.working_app_ids["image_to_3d"] = self.image_to_3d_app_id
            return self.image_to_3d_app_id
            
        # Try default IDs
        for app_id in self.default_app_ids["image_to_3d"]:
            if self._check_app_id(app_id):
                self.working_app_ids["image_to_3d"] = app_id
                return app_id
                
        # No working app ID found
        logging.warning("No working image-to-3D app ID found. Using mock service.")
        return ""
        
    def _check_app_id(self, app_id: str) -> bool:
        """
        Check if an app ID is valid and reachable
        
        Args:
            app_id (str): The app ID to check
            
        Returns:
            bool: True if the app ID is valid and reachable, False otherwise
        """
        if not app_id:
            return False
            
        try:
            # Try to get the manifest for the app ID with a short timeout
            url = f"https://{app_id}/manifest"
            response = requests.get(url, timeout=3)
            return response.status_code == 200
        except Exception as e:
            logging.debug(f"App ID {app_id} check failed: {e}")
            return False
            
    def should_use_mock_service(self, service_type: str) -> bool:
        """
        Determine if a mock service should be used for the given service type
        
        Args:
            service_type (str): The type of service ("text_to_image" or "image_to_3d")
            
        Returns:
            bool: True if a mock service should be used, False if a real service is available
        """
        if service_type == "text_to_image":
            return not self.get_text_to_image_app_id()
        elif service_type == "image_to_3d":
            return not self.get_image_to_3d_app_id()
        else:
            # Unknown service type, use mock service
            return True