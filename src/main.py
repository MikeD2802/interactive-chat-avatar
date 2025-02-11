from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import gradio as gr
from gtts import gTTS
import os
import json
from pathlib import Path
from PIL import Image
import numpy as np
from transformers import pipeline
import imageio  # Ensure this is imported at the top

from avatar_animation import AvatarAnimator
from live_portrait_integration import setup_live_portrait

class ChatAvatar:
    def __init__(self, source_image_path):
        # Initialize updated Ollama LLM
        self.llm = OllamaLLM(model="llama2")
        
        # Initialize conversation history
        self.history = []
        
        # Define avatar personality prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and friendly AI assistant. Respond naturally and conversationally."),
            ("user", "{input}")
        ])
        
        # Use a specific sentiment analysis model
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english",
            revision="714eb0f"
        )
        
        # Verify and load source image
        print(f"Loading source image from: {source_image_path}")
        if not Path(source_image_path).exists():
            raise FileNotFoundError(f"Source image not found at {source_image_path}")
        
        self.source_image = Image.open(source_image_path)
        print(f"Source image loaded. Size: {self.source_image.size}, Mode: {self.source_image.mode}")
        
        # Setup LivePortrait and animator
        self.portrait = setup_live_portrait()
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
        # Debug: print the input text for which we are animating
        print("Animating response for text:", text)
        
        # Analyze sentiment and generate expression parameters
        sentiment = self.sentiment_analyzer(text)
        expression = self.animator.generate_expression(sentiment[0])
        print("Expression parameters:", expression)
        
        # Call LivePortrait integration to generate frames
        frames = self.portrait.generate_animation(self.source_image, expression_params=expression)
        
        if frames is None:
            print("generate_animation returned None!")
            return None
        
        if not isinstance(frames, list):
            print("generate_animation did not return a list. Got:", type(frames))
            return None
        
        print("Number of frames returned:", len(frames))
        
        if len(frames) == 0:
            print("No frames were generated. Check your LivePortrait integration!")
            return None
        
        # Convert the list of frames into a video file (animation.mp4)
        video_path = "animation.mp4"
        try:
            imageio.mimsave(video_path, frames, fps=30)
            print("Video saved to", video_path)
            return video_path
        except Exception as e:
            print("Error creating video:", e)
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
        
        # Print debug information
        print("Response:", response)
        print("Audio file:", audio_file)
        print("Animation file:", animation)
        
        # Return dictionary with expected keys
        return {
            "response": response, 
            "audio": audio_file, 
            "animation": animation
        }
    
    # Create Gradio interface with avatar display
    with gr.Blocks() as iface:
        with gr.Row():
            with gr.Column(scale=2):
                # Explicitly set the type to "messages"
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
    interface.launch(debug=True)  # Added debug mode for more verbose output