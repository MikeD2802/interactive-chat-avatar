import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM  # Updated import
from gtts import gTTS
import os
import json
from pathlib import Path
from PIL import Image
import numpy as np

from avatar_animation import AvatarAnimator
from live_portrait_integration import setup_live_portrait

class ChatAvatar:
    def __init__(self, source_image_path):
        # Initialize Ollama LLM
        self.llm = OllamaLLM(model="llama2")
        
        # Initialize conversation history
        self.history = []
        
        # Define avatar personality prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and friendly AI assistant. Respond naturally and conversationally."),
            ("user", "{input}")
        ])
        
        # Load and setup LivePortrait
        self.portrait = setup_live_portrait()
        self.source_image = Image.open(source_image_path)
        
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
        tts = gTTS(text=text, lang='en')
        tts.save("response.mp3")
        return "response.mp3"
    
    def animate_response(self, text):
        # Analyze sentiment and generate expression parameters
        sentiment = self.animator.analyze_sentiment(text)
        expression = self.animator.generate_expression(sentiment)
        
        # Generate animation using LivePortrait
        frames = self.portrait.generate_animation(
            self.source_image,
            expression_params=expression
        )
        
        return frames

def create_interface(source_image_path):
    avatar = ChatAvatar(source_image_path)
    
    def chat(message, history):
        # Generate text response
        response = avatar.generate_response(message)
        
        # Generate speech
        audio_file = avatar.text_to_speech(response)
        
        # Generate animation
        animation = avatar.animate_response(response)
        
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
        exit(1)
    
    interface = create_interface(source_image)
    interface.launch()