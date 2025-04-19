import os
import logging
import json
from typing import Dict, Any, Optional
import torch

# Conditional imports to handle different model types
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    logging.warning("Transformers library not available, falling back to simpler models")

try:
    import subprocess
    subprocess.run(["pip", "show", "openfabric-pysdk"], check=True)
    import ctransformers
    CTRANSFORMERS_AVAILABLE = True
except ImportError:
    CTRANSFORMERS_AVAILABLE = False
    logging.warning("CTransformers library not available for GGUF models")

class LLMManager:
    """
    Manages a local LLM for prompt enhancement and creative expansion.
    Supports multiple model types and formats based on available resources.
    """
    def __init__(self, model_name="TheBloke/Llama-2-7B-Chat-GGUF", device="cpu"):
        """
        Initialize the LLM Manager with a specified model.
        
        Args:
            model_name (str): HuggingFace model identifier or path to local model
            device (str): Device to run the model on ('cpu', 'cuda', or 'auto')
        """
        self.model_name = model_name
        # Set device automatically based on availability if 'auto'
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        self.model = None
        self.tokenizer = None
        self.pipe = None
        self.initialized = False
        
        # Load templates
        self.templates = {
            "enhance": "You are an expert in generating detailed image descriptions. Take the user's input and enhance it with vivid details, artistic style, lighting, perspective, and mood. Keep your response concise and focused on visual elements only. Don't add explanations or questions.\n\nUser input: {prompt}\n\nEnhanced description:"
        }
        
        # Try to load config if available
        try:
            config_path = "config/llm_config.json"
            if os.path.exists(config_path):
                with open(config_path, 'r') as f:
                    config = json.load(f)
                    if "prompt_templates" in config:
                        self.templates.update(config["prompt_templates"])
        except Exception as e:
            logging.warning(f"Error loading LLM templates: {e}")
        
    def initialize(self):
        """Load the model and tokenizer if not already initialized."""
        if not self.initialized:
            try:
                logging.info(f"Loading LLM model: {self.model_name} on {self.device}")
                
                # Handle different model types
                if ".gguf" in self.model_name.lower() or "gguf" in self.model_name.lower():
                    self._initialize_gguf_model()
                else:
                    self._initialize_transformers_model()
                    
                self.initialized = True
                logging.info("LLM model loaded successfully")
            except Exception as e:
                logging.error(f"Failed to initialize LLM: {e}", exc_info=True)
                # Fall back to a simpler model if available
                self._initialize_fallback_model()
    
    def _initialize_gguf_model(self):
        """Initialize a GGUF quantized model using CTransformers."""
        if not CTRANSFORMERS_AVAILABLE:
            raise ImportError("CTransformers library required for GGUF models")
            
        try:
            # For local GGUF file
            if os.path.exists(self.model_name):
                model_path = self.model_name
            else:
                # For HuggingFace model ID, download if needed
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(
                    repo_id=self.model_name,
                    filename="model.gguf"
                )
                
            # Initialize with ctransformers
            self.model = ctransformers.AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="llama",
                max_new_tokens=512,
                context_length=2048,
                gpu_layers=0 if self.device == "cpu" else 50
            )
            
            # Create a simple pipeline function
            def generate_text(prompt, max_new_tokens=256, temperature=0.7):
                return self.model(prompt, max_new_tokens=max_new_tokens, temperature=temperature)
                
            self.pipe = generate_text
            
        except Exception as e:
            logging.error(f"Failed to initialize GGUF model: {e}", exc_info=True)
            raise
    
    def _initialize_transformers_model(self):
        """Initialize a model using HuggingFace Transformers."""
        if not TRANSFORMERS_AVAILABLE:
            raise ImportError("Transformers library required")
            
        try:
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            
            # Determine quantization based on available RAM
            import psutil
            available_ram_gb = psutil.virtual_memory().available / (1024 ** 3)
            logging.info(f"Available RAM: {available_ram_gb:.2f} GB")
            
            load_in_8bit = available_ram_gb < 12  # Use 8-bit quantization if less than 12GB available
            load_in_4bit = available_ram_gb < 8   # Use 4-bit quantization if less than 8GB available
            
            # Load model with appropriate quantization
            if load_in_4bit:
                logging.info("Using 4-bit quantization for model")
                try:
                    try:
                        import bitsandbytes as bnb
                        self.model = AutoModelForCausalLM.from_pretrained(
                            self.model_name,
                            device_map="auto",
                            load_in_4bit=True,
                            bnb_4bit_compute_dtype=torch.float16
                        )
                    except ImportError:
                        logging.warning("bitsandbytes not available, skipping 4-bit quantization")
                        load_in_4bit = False
                except ImportError:
                    logging.warning("bitsandbytes not available, falling back to 8-bit")
                    load_in_4bit = False
                    load_in_8bit = True
            
            if load_in_8bit and not load_in_4bit:
                logging.info("Using 8-bit quantization for model")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto",
                    load_in_8bit=True
                )
            elif not load_in_4bit:
                logging.info("Loading model in full precision")
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    device_map="auto"
                )
            
            # Create pipeline
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=512,
                device_map="auto"
            )
            
        except Exception as e:
            logging.error(f"Failed to initialize Transformers model: {e}", exc_info=True)
            raise
    
    def _initialize_fallback_model(self):
        """Initialize a very small fallback model when others fail."""
        try:
            logging.warning("Initializing fallback model (GPT-2 small)")
            
            if not TRANSFORMERS_AVAILABLE:
                raise ImportError("Transformers library required even for fallback")
                
            self.model_name = "distilgpt2"  # Very small model that should run anywhere
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(self.model_name)
            
            self.pipe = pipeline(
                "text-generation",
                model=self.model,
                tokenizer=self.tokenizer,
                max_length=256,
                device_map="auto"
            )
            
            self.initialized = True
            logging.info("Fallback model initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize fallback model: {e}", exc_info=True)
            raise
    
    def enhance_prompt(self, prompt: str) -> str:
        """
        Enhance a user prompt with additional details and artistic direction.
        
        Args:
            prompt (str): The original user prompt
            
        Returns:
            str: Enhanced prompt with additional details
        """
        try:
            if not self.initialized:
                self.initialize()
                
            # Get the enhancement template
            template = self.templates.get("enhance", "")
            input_text = template.format(prompt=prompt)
            
            # Generate enhanced text
            if self.pipe:
                output = self.pipe(input_text, max_new_tokens=256, temperature=0.7)
                if isinstance(output, list):
                    generated_text = output[0]['generated_text']
                else:
                    generated_text = output
                    
                # Extract the enhanced part
                enhanced_prompt = generated_text[len(input_text):].strip()
                if not enhanced_prompt:
                    # Fallback if enhancement is empty
                    enhanced_prompt = f"{prompt} with dramatic lighting, detailed textures, and vibrant colors"
                return enhanced_prompt
            else:
                # Fallback if model is not available
                return f"{prompt} with dramatic lighting, detailed textures, and vibrant colors"
                
        except Exception as e:
            logging.error(f"Error enhancing prompt: {e}", exc_info=True)
            # Fallback prompt
            return f"{prompt} with dramatic lighting, detailed textures, and vibrant colors"