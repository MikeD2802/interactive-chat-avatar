# Interactive Chat Avatar

An interactive chat avatar powered by Llama LLM with live portrait AI integration. This project combines live portrait animation with natural language processing to create an engaging conversational AI avatar.

## Features

- Real-time chat interface using Gradio
- LLM-powered responses using Llama
- Text-to-speech conversion
- Sentiment-based avatar animations
- Live portrait integration

## Setup

1. Clone the repository:
```bash
git clone https://github.com/MikeD2802/interactive-chat-avatar.git
cd interactive-chat-avatar
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install Ollama and download the Llama model:
```bash
# Follow instructions at ollama.ai to install Ollama
ollama pull llama2
```

4. Run the application:
```bash
python src/main.py
```

## Project Structure

- `src/main.py`: Core application logic and chat interface
- `src/avatar_animation.py`: Avatar animation and sentiment analysis
- `requirements.txt`: Project dependencies

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License