# Large Language and Visual Assistant for
User-Centric Cybersecurity

A sophisticated AI-powered system for detecting and analyzing potential scam content in images and text using advanced language models and computer vision techniques.

## Features

- Real-time image and text analysis for scam detection
- Integration with Gemma 3:4B language model for advanced content understanding
- OCR capabilities for text extraction from images
- Configurable evaluation pipeline with customizable prompts
- Comprehensive logging and evaluation metrics
- Support for both image and text-only content analysis

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/CSVA.git
cd CSVA
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Install Ollama (required for Gemma model):
```bash
# Follow instructions at https://ollama.ai/download
```

5. Pull the Gemma model:
```bash
ollama pull gemma:3b
```

## Configuration

1. Review and modify `config.yaml` to adjust:
   - Model parameters
   - Evaluation prompts
   - Logging settings
   - Processing thresholds

## Usage

1. Run the main evaluation pipeline:
```bash
python main.py
```

2. For testing and evaluation:
```bash
python test.py
```

## Project Structure

- `core/`: Core evaluation pipeline components
- `processors/`: Image and text processing modules
- `llms/`: Language model integration
- `utils/`: Utility functions and helpers
- `data/`: Data storage and processing
- `logs/`: Evaluation and processing logs
- `captures/`: Screenshot storage
- `captions/`: Image caption storage

## Performance Optimization

The system is optimized for:
- Efficient image processing
- Parallel content analysis
- Memory-efficient model loading
- Configurable processing parameters

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

[Your chosen license]

## Contact

[Your contact information] 