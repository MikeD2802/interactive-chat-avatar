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

class ChatAvatar:
    def __init__(self, source_image_path):
        print("\n=== Initializing ChatAvatar ===")
        # Load environment variables
        load_dotenv()
        print("Loaded environment variables")
        
        try:
            # Initialize LLM
            print("Initializing LLM...")
            self.llm = OllamaLLM(model="llama2")
            print("LLM initialized successfully")
        except Exception as e:
            print(f"Error initializing LLM: {e}")
            traceback.print_exc()
            raise
        
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
            response = self.llm.invoke(message)
            print(f"Generated response: {response}")
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            traceback.print_exc()
            return "I apologize, but I'm having trouble generating a response."

    def text_to_speech(self, text):
        """Convert text to speech using ElevenLabs."""
        print(f"\n=== Converting to speech: {text[:50]}... ===")
        try:
            api_key = os.getenv('ELEVENLABS_API_KEY')
            if not api_key:
                raise ValueError("ElevenLabs API key not found in environment variables")
            print("Found ElevenLabs API key")
            
            print("Generating audio...")
            audio = generate(
                text=text,
                api_key=api_key,
                voice="Rachel"
            )
            print("Audio generated successfully")
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
                temp_audio.write(audio)
                print(f"Audio saved to: {temp_audio.name}")
                return temp_audio.name
        
        except Exception as e:
            print(f"Error generating speech: {e}")
            traceback.print_exc()
            return None

    def animate_response(self, text):
        """Generate animated response."""
        print(f"\n=== Animating response: {text[:50]}... ===")
        try:
            # Analyze sentiment
            print("Analyzing sentiment...")
            sentiment = self.sentiment_analyzer(text)[0]
            print(f"Sentiment analysis result: {sentiment}")
            
            frames = []
            print("Generating animation frames...")
            
            # Process frame
            print("Processing source image...")
            result = self.face_processor.process_frame(self.source_image)
            
            if result is None:
                print("Error: Face processor returned no result")
                return None
                
            if not result.get('face_detected'):
                print("Error: No face detected in processed frame")
                return None
            
            print("Generating animation frame...")
            frame = self.face_processor.animate_frame(
                self.source_image,
                result['landmarks']
            )
            
            if frame is not None:
                frames.append(frame)
                print(f"Generated frame {len(frames)}")
            
            if not frames:
                print("Error: No frames were generated")
                return None
            
            # Save animation
            print("Saving animation...")
            output_path = "animation.mp4"
            imageio.mimsave(
                output_path,
                frames,
                fps=30,
                quality=8,
                macro_block_size=None
            )
            
            print(f"Animation saved to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error in animate_response: {e}")
            traceback.print_exc()
            return None

def create_interface():
    """Create the Gradio interface."""
    print("\n=== Creating Gradio Interface ===")
    try:
        avatar = ChatAvatar("assets/source_image.jpg")
    except Exception as e:
        print(f"Error initializing avatar: {e}")
        traceback.print_exc()
        return None
        
    def chat(message, history):
        """Handle chat messages and generate responses."""
        print(f"\n=== Processing chat message: {message} ===")
        try:
            # Generate response
            response = avatar.generate_response(message)
            print(f"Generated response: {response}")
            
            # Generate speech and animation
            print("Generating speech and animation...")
            audio_file = avatar.text_to_speech(response)
            animation = avatar.animate_response(response)
            
            print(f"Processing complete. Audio: {audio_file}, Animation: {animation}")
            
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": response})
            
            return history, animation, audio_file
            
        except Exception as e:
            print(f"Error in chat function: {e}")
            traceback.print_exc()
            history.append({"role": "user", "content": message})
            history.append({"role": "assistant", "content": "I apologize, but I encountered an error."})
            return history, None, None
    
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