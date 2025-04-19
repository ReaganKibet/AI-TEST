# app/mock_image_service.py
import logging
import os
import random
import io
from PIL import Image, ImageDraw, ImageFont
import numpy as np

class MockImageGenerator:
    """
    A mock image generator that creates placeholder images when external services are unavailable.
    This helps avoid runtime errors when the real image services can't be reached.
    """
    
    def __init__(self):
        """Initialize the mock image generator"""
        self.output_dir = os.path.join("outputs", "images")
        os.makedirs(self.output_dir, exist_ok=True)
        
    def generate_image(self, prompt: str) -> str:
        """
        Generate a mock image based on the text prompt
        
        Args:
            prompt (str): Text description of the image to generate
            
        Returns:
            str: Path to the generated image file
        """
        try:
            logging.info(f"Generating mock image for prompt: {prompt}")
            
            # Generate a simple colored gradient image with the prompt text
            width, height = 512, 512
            
            # Create a simple gradient background
            array = np.zeros([height, width, 3], dtype=np.uint8)
            
            # Use hash of prompt to get consistent colors for the same prompt
            color_seed = hash(prompt) % 1000
            random.seed(color_seed)
            
            # Generate colors based on prompt
            r_start = random.randint(0, 200)
            g_start = random.randint(0, 200)
            b_start = random.randint(0, 200)
            
            r_end = random.randint(r_start, 255)
            g_end = random.randint(g_start, 255)
            b_end = random.randint(b_start, 255)
            
            # Create gradient
            for i in range(height):
                r = int(r_start + (r_end - r_start) * i / height)
                g = int(g_start + (g_end - g_start) * i / height)
                b = int(b_start + (b_end - b_start) * i / height)
                array[i, :, 0] = r
                array[i, :, 1] = g
                array[i, :, 2] = b
            
            # Convert to PIL Image
            img = Image.fromarray(array)
            draw = ImageDraw.Draw(img)
            
            # Add text - prompt summary
            try:
                # Try to load a font, fall back to default if unavailable
                # font = ImageFont.truetype("arial.ttf", 20)
                # Use default font if specific font is unavailable
                draw.text((20, 20), "MOCK IMAGE", fill=(255, 255, 255))
                
                # Add prompt text, wrapped to fit the image
                words = prompt.split()
                lines = []
                current_line = ""
                for word in words:
                    test_line = current_line + " " + word if current_line else word
                    if len(test_line) <= 40:  # Limit line length
                        current_line = test_line
                    else:
                        lines.append(current_line)
                        current_line = word
                if current_line:
                    lines.append(current_line)
                
                # Draw text lines
                y_pos = 50
                for line in lines:
                    draw.text((20, y_pos), line, fill=(255, 255, 255))
                    y_pos += 25
                    
            except Exception as e:
                logging.error(f"Error adding text to mock image: {e}")
            
            # Save the generated image
            filename = f"mock_image_{hash(prompt) % 10000:04d}.png"
            output_path = os.path.join(self.output_dir, filename)
            img.save(output_path)
            
            logging.info(f"Mock image saved to {output_path}")
            return output_path
            
        except Exception as e:
            logging.error(f"Error generating mock image: {e}", exc_info=True)
            # Create an absolute minimal fallback image if everything else fails
            try:
                simple_img = Image.new('RGB', (512, 512), color=(73, 109, 137))
                filename = f"fallback_image_{random.randint(1000, 9999)}.png"
                output_path = os.path.join(self.output_dir, filename)
                simple_img.save(output_path)
                return output_path
            except Exception as e2:
                logging.error(f"Failed to create even a fallback image: {e2}")
                return ""