import os
import logging
import time
import json
import uuid
from typing import Dict, Any, Optional, List
from datetime import datetime

from llm_manager import LLMManager
from app.image_manager import ImageManager
from app.memory_manager import MemoryManager
from core.stub import Stub
# Add these imports at the top of your file:
from mock_image_service import MockImageGenerator
from app_id_handler import AppIDManager

class PipelineManager:
    """
    Orchestrates the complete pipeline from prompt to 3D model,
    integrating all components.
    """
    def __init__(self, stub: Stub, llm: LLMManager, image_mgr: ImageManager, memory_mgr: MemoryManager):
        """
        Initialize the PipelineManager with all required components.
        
        Args:
            stub (Stub): Stub instance for Openfabric API connections
            llm (LLMManager): LLM manager for prompt enhancement
            image_mgr (ImageManager): Image manager for handling images
            memory_mgr (MemoryManager): Memory manager for persistence
        """
        self.stub = stub
        self.llm = llm
        self.image_mgr = image_mgr
        self.memory_mgr = memory_mgr
        self.mock_image_generator = MockImageGenerator()
        self.app_id_manager = AppIDManager()
        
        # Set app IDs for text-to-image and image-to-3D
        self.text_to_image_app_id = "f0997a01-d6d3-a5fe-53d8-561300318557"
        self.image_to_3d_app_id = "69543f29-4d41-4afc-7f29-3d51591f11eb"
        
    def process_prompt(self, prompt: str, session_id: str = "default") -> Dict[str, Any]:
        """
        Process a user prompt through the complete pipeline.
        
        Args:
            prompt (str): User prompt
            session_id (str): Session identifier
            
        Returns:
            Dict[str, Any]: Results including paths to generated assets
        """
        creation_id = str(uuid.uuid4())
        timestamp = datetime.now().isoformat()
        
        try:
            # First, check memory for similar previous creations
            similar_creations = self.memory_mgr.search_by_prompt(prompt)
            
            # Store original prompt in session memory
            self.memory_mgr.store_session_data(session_id, "last_prompt", prompt)
            
            # Step 1: Enhance prompt with LLM
            logging.info(f"Enhancing prompt: {prompt}")
            enhanced_prompt = self.llm.enhance_prompt(prompt)
            self.memory_mgr.store_session_data(session_id, "last_enhanced_prompt", enhanced_prompt)
            
            # Step 2: Generate image from enhanced prompt
            logging.info(f"Generating image from prompt: {enhanced_prompt}")
            image_result = self._generate_image(enhanced_prompt)
            
            if not image_result:
                raise Exception("Failed to generate image")
            
            # Save the generated image
            image_metadata = self.image_mgr.save_image_from_base64(
                image_result.get("image", ""), 
                enhanced_prompt
            )
            
            # Step 3: Generate 3D model from image
            logging.info(f"Generating 3D model from image")
            model_result = self._generate_3d_model(image_metadata["image_path"])
            
            model_path = None
            if model_result and "model" in model_result:
                # Save 3D model (implementation depends on format)
                model_path = os.path.join("outputs", "models", f"{creation_id}.glb")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                
                # Save model data (format depends on Openfabric output)
                with open(model_path, "wb") as f:
                    f.write(model_result["model"])
            
            # Step 4: Store everything in memory
            creation_data = {
                "id": creation_id,
                "timestamp": timestamp,
                "prompt": prompt,
                "enhanced_prompt": enhanced_prompt,
                "image_path": image_metadata["image_path"],
                "thumbnail_path": image_metadata["thumbnail_path"],
                "model_path": model_path,
                "metadata": {
                    "similar_creations": [c["id"] for c in similar_creations],
                    "image_width": image_metadata["width"],
                    "image_height": image_metadata["height"],
                    "processing_time": time.time()
                }
            }
            
            # Store in long-term memory
            self.memory_mgr.store_creation(creation_data)
            
            return creation_data
            
        except Exception as e:
            logging.error(f"Error in pipeline: {e}")
            # Return partial results if any
            return {
                "id": creation_id,
                "timestamp": timestamp,
                "prompt": prompt,
                "error": str(e)
            }
    
    def _generate_image(self, prompt: str, retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Generate image from text using Openfabric API.
        
        Args:
            prompt (str): Text prompt to generate image from
            retries (int): Number of retry attempts
            
        Returns:
            Optional[Dict[str, Any]]: Result containing the generated image
        """
        # Get input schema for text-to-image app
        input_schema = self.stub.schema(self.text_to_image_app_id, 'input')
        
        # Build request based on schema
        request_data = {
            "prompt": prompt,
            "negative_prompt": "blurry, low quality, distorted, watermark",
            "width": 512,
            "height": 512,
            "num_inference_steps": 30,
            "guidance_scale": 7.5
        }
        
        # Try to generate image with retries
        for attempt in range(retries):
            try:
                result = self.stub.call(self.text_to_image_app_id, request_data)
                if result and "image" in result:
                    return result
                logging.warning(f"Attempt {attempt+1}: Image generation failed, retrying...")
                time.sleep(2)
            except Exception as e:
                logging.error(f"Error in image generation (attempt {attempt+1}): {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(2)
        
        return None
    
    def _generate_3d_model(self, image_path: str, retries: int = 3) -> Optional[Dict[str, Any]]:
        """
        Generate 3D model from image using Openfabric API.
        
        Args:
            image_path (str): Path to the input image
            retries (int): Number of retry attempts
            
        Returns:
            Optional[Dict[str, Any]]: Result containing the generated 3D model
        """
        # Convert image to base64
        image_base64 = self.image_mgr.image_to_base64(image_path)
        
        # Get input schema for image-to-3D app
        input_schema = self.stub.schema(self.image_to_3d_app_id, 'input')
        
        # Build request based on schema
        request_data = {
            "image": f"data:image/png;base64,{image_base64}",
            "format": "glb",  # Or other format supported by the API
            "quality": "medium"
        }
        
        # Try to generate 3D model with retries
        for attempt in range(retries):
            try:
                result = self.stub.call(self.image_to_3d_app_id, request_data)
                if result and "model" in result:
                    return result
                logging.warning(f"Attempt {attempt+1}: 3D model generation failed, retrying...")
                time.sleep(2)
            except Exception as e:
                logging.error(f"Error in 3D model generation (attempt {attempt+1}): {e}")
                if attempt == retries - 1:
                    raise
                time.sleep(2)
        
        return None

    def execute(self, prompt: str) -> Dict[str, Any]:
        """
        Execute the creative AI pipeline based on the prompt.
        
        Args:
            prompt (str): The user-provided prompt.
            
        Returns:
            Dict[str, Any]: The result of the pipeline execution.
        """
        try:
            logging.info(f"Received prompt: {prompt}")
            
            # Get app IDs from AppIDManager
            text_to_image_app_id = self.app_id_manager.get_text_to_image_app_id() or self.text_to_image_app_id
            image_to_3d_app_id = self.app_id_manager.get_image_to_3d_app_id() or self.image_to_3d_app_id
            
            # Enhance the prompt using LLM
            logging.info("Enhancing prompt with LLM")
            enhanced_prompt = self.llm.enhance_prompt(prompt)
            
            # Generate image
            logging.info("Generating image")
            try:
                if not self.app_id_manager.should_use_mock_service("text_to_image"):
                    image_result = self._generate_image(enhanced_prompt)
                    if not image_result:
                        raise Exception("Failed to generate image using real service")
                else:
                    raise Exception("Using mock service by design")
            except Exception as e:
                logging.warning(f"Falling back to mock image service: {e}")
                image_path = self.mock_image_generator.generate_image(enhanced_prompt)
                return {
                    "status": "success",
                    "message": "Image generated using mock service",
                    "image_url": image_path,
                    "prompt": enhanced_prompt
                }
            
            # Generate 3D model
            logging.info("Generating 3D model")
            model_result = self._generate_3d_model(image_result["image_path"])
            if not model_result:
                raise Exception("Failed to generate 3D model")
            
            # Return the final result
            return {
                "status": "success",
                "message": "Pipeline executed successfully",
                "image_url": image_result["image"],
                "model_url": model_result["model"],
                "prompt": enhanced_prompt
            }
        
        except Exception as e:
            logging.error(f"Error in pipeline execution: {e}")
            return {
                "status": "error",
                "message": f"An error occurred: {str(e)}",
                "image_url": None,
                "model_url": None,
                "prompt": prompt
            }
