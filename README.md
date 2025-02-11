# Interactive Chat Avatar

An interactive chat avatar powered by LLaMA, SadTalker, and ElevenLabs. This project creates a realistic talking avatar that responds to user input with natural language and facial expressions.

## Features

* Realistic talking avatar using SadTalker
* Natural language responses using LLaMA
* High-quality voice synthesis with ElevenLabs
* Real-time chat interface using Gradio
* Automatic model downloading and setup

## Prerequisites

1. NVIDIA GPU with CUDA support (recommended)
2. Python 3.10
3. Git
4. [Ollama](https://ollama.ai) installed

## Setup

1. Clone the repository:
```bash
git clone https://github.com/MikeD2802/interactive-chat-avatar.git
cd interactive-chat-avatar
```

2. Create and activate a conda environment:
```bash
conda create -n avatar-chat python=3.10
conda activate avatar-chat
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

5. Download LLaMA model:
```bash
ollama pull llama2
```

6. Prepare your avatar source image:
```bash
mkdir -p assets
# Add your image as assets/source_image.jpg
```

## Running the Application

1. Start the application:
```bash
python src/main.py
```

2. Open your web browser and navigate to the URL shown in the terminal (usually http://localhost:7860)

3. Start chatting with your AI avatar!

## Project Structure

* `src/main.py`: Core application and Gradio interface
* `src/avatar_generator.py`: SadTalker integration
* `src/utils/download.py`: Model download utilities
* `requirements.txt`: Project dependencies
* `assets/`: Directory for source images

## How It Works

1. User sends a message through the Gradio interface
2. LLaMA processes the message and generates a response
3. ElevenLabs converts the response to speech
4. SadTalker animates the avatar to match the speech
5. The result is displayed in real-time in the interface

## Customization

1. Avatar Image:
   - Replace `assets/source_image.jpg` with your preferred image
   - Image should be a clear front-facing portrait

2. Voice:
   - Modify the voice ID in `src/main.py` to use different ElevenLabs voices
   - Available voices can be found in your ElevenLabs dashboard

3. Language Model:
   - Different LLaMA models can be used by modifying the model name in `src/main.py`
   - Other Ollama models can be used as well

## Troubleshooting

1. CUDA Issues:
   - Ensure NVIDIA drivers are up to date
   - Check CUDA version compatibility with PyTorch

2. Memory Issues:
   - Reduce batch size in avatar_generator.py
   - Use a smaller LLaMA model

3. Model Downloads:
   - Check internet connection
   - Ensure enough disk space for models

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License

## Acknowledgments

* [SadTalker](https://github.com/OpenTalker/SadTalker) for the talking face animation
* [Ollama](https://ollama.ai/) for the LLM integration
* [ElevenLabs](https://elevenlabs.io/) for voice synthesis
* [Gradio](https://www.gradio.app/) for the web interface