import gradio as gr
from langchain_ollama import OllamaLLM
from real_time_animation import RealTimeAnimator
import cv2
import numpy as np
from transformers import pipeline
import os
import imageio
import tempfile
from elevenlabs import generate
from dotenv import load_dotenv

class ChatAvatar:
    def __init__(self, source_image_path):
        # Load environment variables
        load_dotenv()
        
        # Initialize LLM
        self.llm = OllamaLLM(model="llama2")
        
        # Initialize sentiment analyzer
        self.sentiment_analyzer = pipeline(
            "sentiment-analysis",
            model="distilbert-base-uncased-finetuned-sst-2-english",
            revision="714eb0f"
        )
        
        # Initialize real-time animator
        self.animator = RealTimeAnimator()
        self.source_image_path = source_image_path
        
        # Verify source image exists
        if not os.path.exists(source_image_path):
            raise FileNotFoundError(f"Source image not found: {source_image_path}")
            
        print(f"Initialized ChatAvatar with source image: {source_image_path}")
        
    def generate_response(self, message):
        """Generate a response using the LLM."""
        try:
            response = self.llm.invoke(message)
            print(f"Generated response: {response}")
            return response
        except Exception as e:
            print(f"Error generating response: {e}")
            return "I apologize, but I'm having trouble generating a response."
            
    def text_to_speech(self, text):
        """Convert text to speech using ElevenLabs."""
        try:
            api_key = os.getenv('ELEVENLABS_API_KEY')
            if not api_key:
                raise ValueError("ElevenLabs API key not found")
                
            audio = generate(
                text=text,
                api_key=api_key,
                voice="Rachel"
            )
            
            # Save audio to temporary file
            with tempfile.NamedTemporaryFile(suffix='.mp3', delete=False) as temp_audio:
                temp_audio.write(audio)
                return temp_audio.name
                
        except Exception as e:
            print(f"Error generating speech: {e}")
            return None
            
    def animate_response(self, text):
        """Generate animated response with real-time facial animation."""
        try:
            # Load source image
            source_image = cv2.imread(self.source_image_path)
            if source_image is None:
                raise ValueError(f"Failed to load source image: {self.source_image_path}")
                
            print(f"Loaded source image with shape: {source_image.shape}")
            
            # Generate frames with facial animation
            frames = []
            cap = cv2.VideoCapture(0)  # Use webcam for demonstration
            
            frame_count = 0
            while frame_count < 30:  # Generate 30 frames
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame to get facial landmarks
                landmarks = self.animator.process_frame(frame)
                if landmarks is not None:
                    # Animate source image using landmarks
                    animated_frame = self.animator.animate_frame(source_image, landmarks)
                    if animated_frame is not None:
                        frames.append(animated_frame)
                        frame_count += 1
                        
            cap.release()
            
            if not frames:
                print("No frames were generated")
                return None
                
            # Save frames as video
            output_path = "animation.mp4"
            imageio.mimsave(output_path, frames, fps=30)
            print(f"Saved animation to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error in animate_response: {e}")
            return None

def create_interface():
    """Create the Gradio interface."""
    avatar = ChatAvatar("assets/source_image.jpg")
    
    def chat(message, history):
        """Handle chat messages and generate responses."""
        try:
            # Generate response
            response = avatar.generate_response(message)
            
            # Generate speech
            audio_file = avatar.text_to_speech(response)
            
            # Generate animation
            animation = avatar.animate_response(response)
            
            print(f"Response: {response}")
            print(f"Audio file: {audio_file}")
            print(f"Animation file: {animation}")
            
            return {
                "response": response,
                "audio": audio_file,
                "animation": animation
            }
            
        except Exception as e:
            print(f"Error in chat function: {e}")
            return {
                "response": "I apologize, but I encountered an error.",
                "audio": None,
                "animation": None
            }
    
    with gr.Blocks() as demo:
        chatbot = gr.Chatbot(type="messages")
        video = gr.Video()
        audio = gr.Audio()
        
        msg = gr.Textbox()
        send = gr.Button("Send")
        
        send.click(chat, [msg, chatbot], [chatbot, video, audio])
        
    return demo

if __name__ == "__main__":
    demo = create_interface()
    demo.launch(debug=True)
