import gradio as gr
import logging
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import Ollama
from gtts import gTTS
import os
import json
from pathlib import Path
from PIL import Image
import numpy as np

from avatar_animation import AvatarAnimator
from live_portrait_integration import setup_live_portrait, generate_avatar_animation

# Configure logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

class ChatAvatar:
    def __init__(self, source_image_path):
        # Initialize Ollama LLM
        self.llm = Ollama(model="llama2")
        
        # Initialize conversation history
        self.history = []
        
        # Define avatar personality prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and friendly AI assistant. Respond naturally and conversationally."),
            ("user", "{input}")
        ])
        
        # Load and setup LivePortrait
        self.portrait = setup_live_portrait()
        
        # Load source image
        try:
            self.source_image = Image.open(source_image_path)
        except Exception as e:
            logging.error(f"Failed to load source image: {e}")
            self.source_image = None
        
        # Initialize sentiment analyzer for expressions
        self.animator = AvatarAnimator()
    
    def generate_response(self, user_input):
        # Add user input to history
        self.history.append({"role": "user", "content": user_input})
        
        # Generate response using LLM
        response = self.llm.invoke(
            self.prompt.format(input=user_input)
        )
        
        # Add response to history
        self.history.append({"role": "assistant", "content": response})
        
        return response
    
    def text_to_speech(self, text):
        # Convert text to speech
        try:
            tts = gTTS(text=text, lang='en')
            tts.save("response.mp3")
            return "response.mp3"
        except Exception as e:
            logging.error(f"Text-to-speech conversion failed: {e}")
            return None
    
    def animate_response(self, text):
        # Verify source image and portrait model
        if self.source_image is None or self.portrait is None:
            logging.warning("Cannot generate animation: missing source image or portrait model")
            return None
        
        # Analyze sentiment and generate expression parameters
        sentiment = self.animator.analyze_sentiment(text)
        expression = self.animator.generate_expression(sentiment)
        
        # Generate animation using LivePortrait
        try:
            frames = generate_avatar_animation(
                self.portrait, 
                self.source_image, 
                expression_params=expression
            )
            return frames
        except Exception as e:
            logging.error(f"Avatar animation generation failed: {e}")
            return None

def create_interface(source_image_path):
    avatar = ChatAvatar(source_image_path)
    
    def chat(message, history):
        # Generate text response
        response = avatar.generate_response(message)
        
        # Generate speech
        audio_file = avatar.text_to_speech(response)
        
        # Generate animation
        animation = avatar.animate_response(response)
        
        # Handle potential None values
        audio_file = audio_file or ''
        animation = animation or np.zeros((100, 100, 3), dtype=np.uint8)  # Placeholder black frame
        
        return {
            "response": response,
            "audio": audio_file,
            "animation": animation
        }
    
    # Create Gradio interface with avatar display
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(type="messages")
                msg = gr.Textbox(label="Message")
                clear = gr.ClearButton([msg, chatbot])
                
            with gr.Column(scale=1):
                avatar_video = gr.Video(label="Avatar")
                audio_output = gr.Audio(label="Response Audio")
        
        msg.submit(
            chat,
            [msg, chatbot],
            [chatbot, avatar_video, audio_output]
        )
    
    return iface

if __name__ == "__main__":
    # Default source image path
    source_image = "assets/source_image.jpg"
    
    # Create assets directory if it doesn't exist
    Path("assets").mkdir(exist_ok=True)
    
    # Check if source image exists
    if not Path(source_image).exists():
        print(f"Please place a source image at {source_image}")
        print("You can use any clear, frontal portrait image.")
        exit(1)
    
    interface = create_interface(source_image)
    interface.launch()