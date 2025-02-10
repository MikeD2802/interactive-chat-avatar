# Interactive Chat Avatar

An interactive chat avatar powered by Llama LLM with live portrait AI integration. This project combines live portrait animation with natural language processing to create an engaging conversational AI avatar.

## Features

- Real-time chat interface using Gradio
- LLM-powered responses using Llama
- Text-to-speech conversion
- Sentiment-based avatar animations using LivePortrait
- Live portrait integration for realistic facial expressions

## Prerequisites

1. NVIDIA GPU with CUDA support (recommended)
2. Python 3.10
3. Git

## Setup

1. Clone the repository and LivePortrait:
```bash
# Clone main repository
git clone https://github.com/MikeD2802/interactive-chat-avatar.git
cd interactive-chat-avatar

# Clone LivePortrait
git clone https://github.com/KwaiVGI/LivePortrait
```

2. Create and activate a conda environment:
```bash
conda create -n chat-avatar python=3.10
conda activate chat-avatar
```

3. Install PyTorch with CUDA support:
```bash
# For CUDA 11.8 (recommended)
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 --index-url https://download.pytorch.org/whl/cu118
```

4. Install other dependencies:
```bash
pip install -r requirements.txt
```

5. Download LivePortrait pretrained weights:
```bash
pip install -U "huggingface_hub[cli]"
huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights --exclude "*.git*" "README.md" "docs"
```

6. Install Ollama and download the Llama model:
```bash
# Follow instructions at ollama.ai to install Ollama
ollama pull llama2
```

7. Prepare your avatar source image:
```bash
# Place your desired avatar image in the assets directory
mkdir -p assets
# Copy your image to assets/source_image.jpg
```

## Running the Application

1. Start the application:
```bash
python src/main.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:7860)

3. Start chatting with your AI avatar!

## Project Structure

- `src/main.py`: Core application logic and chat interface
- `src/avatar_animation.py`: Avatar animation and sentiment analysis
- `src/live_portrait_integration.py`: LivePortrait model integration
- `requirements.txt`: Project dependencies
- `assets/`: Directory for source images and other assets
- `pretrained_weights/`: Directory for LivePortrait model weights

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

- [LivePortrait](https://github.com/KwaiVGI/LivePortrait) for the portrait animation technology
- [Ollama](https://ollama.ai/) for the LLM integration
- [Gradio](https://www.gradio.app/) for the web interface