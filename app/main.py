import logging
import os
import json
from typing import Dict

from ontology_dc8f06af066e4a7880a5938933236037.config import ConfigClass
from ontology_dc8f06af066e4a7880a5938933236037.input import InputClass
from ontology_dc8f06af066e4a7880a5938933236037.output import OutputClass
from openfabric_pysdk.context import AppModel, State
from core.stub import Stub

# Import configuration
from config.logging_config import configure_logging

# Import our custom managers
from llm_manager import LLMManager
from image_manager import ImageManager
from memory_manager import MemoryManager
from pipeline_manager import PipelineManager
from utils import load_json_config

# Import Flask and extensions for Swagger UI
from flask import Flask, request, jsonify, send_from_directory
from flask_apispec import FlaskApiSpec, doc, use_kwargs, marshal_with
from apispec import APISpec
from apispec.ext.marshmallow import MarshmallowPlugin
from marshmallow import fields, Schema

# Configure logging
configure_logging()

# Ensure output directories exist
os.makedirs("outputs", exist_ok=True)
os.makedirs("outputs/images", exist_ok=True)
os.makedirs("outputs/models", exist_ok=True)
os.makedirs("outputs/metadata", exist_ok=True)
os.makedirs("outputs/thumbnails", exist_ok=True)

# Load LLM configuration
llm_config = load_json_config("config/llm_config.json")

# Configurations for the app
configurations: Dict[str, ConfigClass] = dict()

# Initialize managers (lazy loading)
llm_manager = LLMManager(
    model_name=llm_config.get("model_name", "TheBloke/Llama-2-7B-Chat-GGUF"),
    device=llm_config.get("device", "cpu")
)
image_manager = ImageManager(output_dir="outputs")
memory_manager = MemoryManager(db_path="outputs/memory.db")

# Default app IDs (will be overridden by user config if provided)
DEFAULT_APP_IDS = [
    "f0997a01-d6d3-a5fe-53d8-561300318557",  # Text-to-Image
    "69543f29-4d41-4afc-7f29-3d51591f11eb"   # Image-to-3D
]

############################################################
# Config callback function
############################################################
def config(configuration: Dict[str, ConfigClass], state: State) -> None:
    """
    Stores user-specific configuration data.

    Args:
        configuration (Dict[str, ConfigClass]): A mapping of user IDs to configuration objects.
        state (State): The current state of the application.
    """
    for uid, conf in configuration.items():
        logging.info(f"Saving new config for user with id:'{uid}'")
        configurations[uid] = conf


############################################################
# Execution callback function
############################################################
def execute(model: AppModel) -> None:
    """
    Main execution entry point for handling a model pass.

    Args:
        model (AppModel): The model object containing request and response structures.
    """
    # Retrieve input
    request: InputClass = model.request
    user_prompt = request.prompt
    
    logging.info(f"Received prompt: {user_prompt}")
    
    # Generate a session ID (using user ID or a default)
    session_id = getattr(request, "user_id", "default_session")
    
    # Retrieve user config
    user_config: ConfigClass = configurations.get('super-user', None)
    
    # Initialize the Stub with app IDs
    app_ids = DEFAULT_APP_IDS
    if user_config and hasattr(user_config, 'app_ids') and user_config.app_ids:
        app_ids = user_config.app_ids
        logging.info(f"Using custom app IDs from config: {app_ids}")
    else:
        logging.info(f"Using default app IDs: {app_ids}")
    
    # Create stub with app IDs
    stub = Stub(app_ids)
    
    try:
        # Initialize pipeline manager
        pipeline = PipelineManager(stub, llm_manager, image_manager, memory_manager)
        
        # Check if the prompt is referencing previous creations
        if any(keyword in user_prompt.lower() for keyword in ["like before", "previous", "last time", "like the one"]):
            logging.info("Detected reference to previous creation")
            # Get context from memory
            recent_creations = memory_manager.get_recent_creations(5)
            if recent_creations:
                logging.info(f"Found {len(recent_creations)} recent creations")
                # Use the most recent creation as context
                last_creation = recent_creations[0]
                memory_manager.store_session_data(session_id, "reference_creation", last_creation)
        
        # Process the prompt through the entire pipeline
        logging.info(f"Processing prompt through pipeline")
        result = pipeline.process_prompt(user_prompt, session_id)
        
        # Check if we had an error
        if "error" in result:
            response_message = f"An error occurred: {result['error']}"
            logging.error(f"Pipeline error: {result['error']}")
        else:
            # Format successful result
            response_message = (
                f"Successfully processed your request!\n\n"
                f"Original prompt: {result['prompt']}\n"
                f"Enhanced prompt: {result['enhanced_prompt']}\n\n"
                f"Image saved to: {os.path.basename(result['image_path'])}\n"
            )
            
            if result.get('model_path'):
                response_message += f"3D model saved to: {os.path.basename(result['model_path'])}\n"
            else:
                response_message += "3D model generation did not complete successfully.\n"
                
            # Add memory information
            recent_creations = memory_manager.get_recent_creations(3)
            if len(recent_creations) > 1:  # More than just the current one
                response_message += "\nYour recent creations:\n"
                for i, creation in enumerate(recent_creations):
                    if creation['id'] != result['id']:  # Skip current one
                        response_message += f"- {creation['prompt'][:50]}...\n"
            
            # Add similar creations if any
            similar_ids = result.get('metadata', {}).get('similar_creations', [])
            if similar_ids:
                response_message += "\nSimilar to your previous creations.\n"
                
            logging.info("Pipeline completed successfully")
    
    except Exception as e:
        logging.error(f"Error in execute function: {e}", exc_info=True)
        response_message = f"An error occurred while processing your request: {str(e)}"
    
    # Prepare response
    response: OutputClass = model.response
    response.message = response_message
    logging.info(f"Response prepared: {len(response_message)} characters")

# Create response schema
class ResponseSchema(Schema):
    message = fields.Str(metadata={"description": "Response message"})
    error = fields.Str(metadata={"description": "Error message if any"})

# Create request schema
class PromptSchema(Schema):
    prompt = fields.Str(required=True, metadata={"description": "The prompt to generate from"})

# Initialize Flask app
app = Flask(__name__)

# Initialize Swagger documentation
app.config.update({
    'APISPEC_SPEC': APISpec(
        title='AI Creative Partner API',
        version='1.0.0',
        openapi_version='2.0',
        plugins=[MarshmallowPlugin()],
    ),
    'APISPEC_SWAGGER_URL': '/swagger.json',
    'APISPEC_SWAGGER_UI_URL': '/swagger-ui/',
    'APISPEC_SWAGGER_UI_CONFIG': {
        'deepLinking': True,
        'persistAuthorization': True,
        'displayOperationId': False,
        'defaultModelsExpandDepth': 3,
        'defaultModelExpandDepth': 3,
        'defaultModelRendering': 'model',
        'displayRequestDuration': True,
        'docExpansion': 'list',
        'filter': True,
        'showExtensions': True,
        'showCommonExtensions': True,
        'supportedSubmitMethods': ['get', 'post', 'put', 'delete', 'options', 'head', 'patch', 'trace'],
        'tryItOutEnabled': True
    }
})

# Initialize FlaskApiSpec
docs = FlaskApiSpec(app)

# Add these routes
@app.route("/", methods=["GET"])
def home():
    return """
    <html>
        <head>
            <title>AI Creative Partner</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    max-width: 800px;
                    margin: 0 auto;
                    padding: 20px;
                }
                textarea {
                    width: 100%;
                    height: 100px;
                    margin-bottom: 10px;
                }
                button {
                    padding: 8px 16px;
                    background-color: #4CAF50;
                    color: white;
                    border: none;
                    cursor: pointer;
                }
                #response {
                    margin-top: 20px;
                    white-space: pre-wrap;
                    background-color: #f5f5f5;
                    padding: 15px;
                    border-radius: 5px;
                    min-height: 100px;
                }
            </style>
        </head>
        <body>
            <h1>AI Creative Partner API</h1>
            <div>
                <h3>Enter your prompt:</h3>
                <textarea id="promptInput" placeholder="Create an image of a mountain landscape"></textarea>
                <button onclick="sendRequest()">Submit</button>
            </div>
            <div>
                <h3>Response:</h3>
                <div id="response"></div>
            </div>

            <script>
                function sendRequest() {
                    const prompt = document.getElementById('promptInput').value;
                    const responseDiv = document.getElementById('response');
                    
                    responseDiv.textContent = "Sending request...";
                    
                    fetch('/execute', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ prompt: prompt })
                    })
                    .then(response => response.json())
                    .then(data => {
                        responseDiv.textContent = JSON.stringify(data, null, 2);
                    })
                    .catch(error => {
                        responseDiv.textContent = `Error: ${error}`;
                    });
                }
            </script>
        </body>
    </html>
    """

# Add a route to serve images if needed
@app.route("/outputs/images/<path:filename>")
def serve_image(filename):
    return send_from_directory("outputs/images", filename)

@app.route("/execute", methods=["POST"])
@doc(description='Execute a prompt for creative generation', tags=['App'])
@use_kwargs(PromptSchema, location='json')
@marshal_with(ResponseSchema)
def execute_endpoint(**kwargs):
    """
    Execute a creative generation prompt.
    """
    try:
        # Get the prompt from the request body
        input_data = request.get_json()
        if not input_data or 'prompt' not in input_data:
            return {"error": "No prompt provided"}, 400
            
        # Create input and output objects
        input_obj = InputClass(prompt=input_data["prompt"])
        output_obj = OutputClass()
        
        # Create the model with input and output
        model = AppModel()
        model.request = input_obj
        model.response = output_obj
        
        # Call the execute function
        execute(model)
        
        # Return the response
        return {"message": str(model.response.message)}
    except Exception as e:
        logging.error(f"Error in /execute endpoint: {e}", exc_info=True)
        return {"error": str(e)}, 500

# Register the endpoint with Swagger
docs.register(execute_endpoint)

if __name__ == "__main__":
    # Make sure to bind to 0.0.0.0 to accept external connections
    app.run(host="0.0.0.0", port=8888, debug=True)