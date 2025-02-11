import gradio as gr
from langchain_ollama import OllamaLLM
from face_processor import FaceProcessor
import cv2
import numpy as np
from transformers import pipeline
import os
import imageio
import tempfile
from elevenlabs import generate
from dotenv import load_dotenv
import time
import json
import traceback
import requests

def check_ollama():
    """Check if Ollama is running."""
    try:
        response = requests.get("http://localhost:11434/api/tags")
        return response.status_code == 200
    except:
        return False

class ChatAvatar:
    def __init__(self, source_image_path):
        print("\n=== Initializing ChatAvatar ===")
        # Load environment variables
        load_dotenv()
        print("Loaded environment variables")
        
        # Check Ollama status
        if not check_ollama():
            raise ConnectionError(
                "Ollama is not running. Please start it with 'ollama serve' in a terminal."
            )
        
        try:
            # Initialize LLM
            print("Initializing LLM...")
            self.llm = OllamaLLM(model="llama2")
            # Test the connection
            self.llm.invoke("test")
            print("LLM initialized and tested successfully")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            traceback.print_exc()
            raise

        # Continue with rest of initialization...
        try:
            # Initialize sentiment analyzer
            print("Initializing sentiment analyzer...")
            self.sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="distilbert-base-uncased-finetuned-sst-2-english",
                revision="714eb0f"
            )
            print("Sentiment analyzer initialized successfully")
        except Exception as e:
            print(f"Error initializing sentiment analyzer: {e}")
            traceback.print_exc()
            raise
        
        try:
            # Initialize face processor
            print("Initializing face processor...")
            self.face_processor = FaceProcessor()
            self.source_image_path = source_image_path
            print("Face processor initialized successfully")
        except Exception as e:
            print(f"Error initializing face processor: {e}")
            traceback.print_exc()
            raise
        
        # Load and verify source image
        try:
            print(f"Loading source image from: {source_image_path}")
            if not os.path.exists(source_image_path):
                raise FileNotFoundError(f"Source image not found: {source_image_path}")
            
            self.source_image = cv2.imread(source_image_path)
            if self.source_image is None:
                raise ValueError(f"Failed to load source image: {source_image_path}")
            
            print(f"Source image loaded successfully. Shape: {self.source_image.shape}")
            
            # Verify face detection
            print("Detecting face in source image...")
            source_face = self.face_processor.detect_face(self.source_image)
            if source_face is None:
                raise ValueError("No face detected in source image")
            print("Face detected successfully in source image")
            
        except Exception as e:
            print(f"Error loading source image: {e}")
            traceback.print_exc()
            raise

    def generate_response(self, message):
        """Generate a response using the LLM."""
        print(f"\n=== Generating response for: {message} ===")
        try:
            # Check Ollama status before generating response
            if not check_ollama():
                raise ConnectionError("Ollama is not running")
                
            response = self.llm.invoke(message)
            print(f"Generated response: {response}")
            return response
        except ConnectionError as e:
            print(f"Ollama connection error: {e}")
            return "I apologize, but I'm having trouble connecting to the language model. Please ensure Ollama is running."
        except Exception as e:
            print(f"Error generating response: {e}")
            traceback.print_exc()
            return "I apologize, but I'm having trouble generating a response."

    # ... rest of the class implementation ...

def create_interface():
    """Create the Gradio interface."""
    print("\n=== Creating Gradio Interface ===")
    
    # Check Ollama status before creating interface
    if not check_ollama():
        print("WARNING: Ollama is not running. Please start it with 'ollama serve'")
        gr.Warning("Ollama is not running. Please start it with 'ollama serve'")
    
    try:
        avatar = ChatAvatar("assets/source_image.jpg")
    except Exception as e:
        print(f"Error initializing avatar: {e}")
        
        # Create minimal interface to show error
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            gr.Markdown("# Interactive Chat Avatar")
            gr.Markdown(f"⚠️ Error: {str(e)}")
            gr.Markdown("Please check the terminal for more details.")
        return demo
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Interactive Chat Avatar")
        
        with gr.Row():
            with gr.Column(scale=1):
                video = gr.Video(label="Avatar")
                audio = gr.Audio(label="Response Audio")
                
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(type="messages", height=400)
                with gr.Row():
                    msg = gr.Textbox(
                        placeholder="Type your message...",
                        show_label=False
                    )
                    send = gr.Button("Send")
                    
        with gr.Row():
            clear = gr.Button("Clear Chat")
            
        # Event handlers
        msg.submit(chat, [msg, chatbot], [chatbot, video, audio])
        send.click(chat, [msg, chatbot], [chatbot, video, audio])
        clear.click(lambda: None, None, chatbot)
        
    return demo

if __name__ == "__main__":
    print("\n=== Starting Application ===")
    demo = create_interface()
    if demo:
        print("Launching Gradio interface...")
        demo.launch(debug=True)
    else:
        print("Failed to initialize the interface")