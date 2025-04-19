# AI Creative Generation Platform

A powerful AI platform that combines Large Language Models (LLMs) and image generation capabilities to create an interactive creative generation system. This platform provides a RESTful API interface with Swagger documentation for easy integration and usage.

## 🌟 Features

- **LLM Integration**: Powered by state-of-the-art language models
- **Image Generation**: Advanced image processing and generation capabilities
- **Memory Management**: Sophisticated memory system for context-aware responses
- **Pipeline Management**: Flexible pipeline system for processing requests
- **RESTful API**: Well-documented API endpoints with Swagger UI
- **Docker Support**: Containerized deployment with Docker and Docker Compose
- **Configurable**: Extensive configuration options for customization

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Docker and Docker Compose (for containerized deployment)
- CUDA-capable GPU (optional, for faster inference)

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Create and activate a virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the Application

#### Local Development

1. Start the application:
```bash
python app/main.py
```

2. Access the Swagger UI at `http://localhost:8000`

#### Docker Deployment

1. Build and run using Docker Compose:
```bash
docker-compose up --build
```

2. Access the application at `http://localhost:8000`

## 📚 API Documentation

The API documentation is available through Swagger UI at the root endpoint (`/`). Key endpoints include:

- `POST /execute`: Generate creative content based on prompts
- `GET /outputs/images/<filename>`: Access generated images
- Additional endpoints for configuration and management

## 🛠️ Project Structure

```
.
├── app/                    # Main application directory
│   ├── core/              # Core functionality
│   ├── config/            # Configuration files
│   ├── datastore/         # Data storage
│   ├── main.py            # Main application entry point
│   ├── llm_manager.py     # LLM management
│   ├── image_manager.py   # Image processing
│   ├── memory_manager.py  # Memory system
│   └── pipeline_manager.py # Pipeline management
├── outputs/               # Generated outputs
├── logs/                  # Application logs
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose configuration
└── requirements.txt      # Python dependencies
```

## ⚙️ Configuration

The application can be configured through various configuration files:

- `config/llm_config.json`: LLM model settings
- Environment variables for sensitive configurations
- Docker environment variables for containerized deployment

## 🔧 Development

### Adding New Features

1. Create new modules in the appropriate directories
2. Update the pipeline manager for new processing steps
3. Add new API endpoints in `main.py`
4. Update Swagger documentation

### Testing

```bash
# Run tests
python -m pytest tests/
```


## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request



## 🙏 Acknowledgments

- OpenFabric SDK
- Hugging Face Transformers
- Other open-source libraries and tools used in this project