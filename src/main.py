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
        
        # Initialize face processor
        self.face_processor = FaceProcessor()
        self.source_image_path = source_image_path
        
        # Load and verify source image
        if not os.path.exists(source_image_path):
            raise FileNotFoundError(f"Source image not found: {source_image_path}")
            
        self.source_image = cv2.imread(source_image_path)
        if self.source_image is None:
            raise ValueError(f"Failed to load source image: {source_image_path}")
            
        # Verify face detection in source image
        source_face = self.face_processor.detect_face(self.source_image)
        if source_face is None:
            raise ValueError("No face detected in source image")
            
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
        """Generate animated response using advanced face processing."""
        try:
            # Analyze sentiment to adjust animation style
            sentiment = self.sentiment_analyzer(text)[0]
            print(f"Sentiment: {sentiment}")
            
            # Adjust face processor parameters based on sentiment
            if sentiment['label'] == 'POSITIVE':
                self.face_processor.smooth_factor = 0.7  # Smoother for positive emotions
            else:
                self.face_processor.smooth_factor = 0.5  # Default smoothing
                
            # Generate frames
            frames = []
            cap = cv2.VideoCapture(0)
            
            frame_count = 0
            max_frames = 90  # 3 seconds at 30fps
            
            start_time = time.time()
            while frame_count < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                    
                # Process frame
                result = self.face_processor.process_frame(frame)
                if result is not None:
                    # Animate source image using detected landmarks
                    animated_frame = self.face_processor.animate_frame(
                        self.source_image,
                        result['landmarks']
                    )
                    
                    if animated_frame is not None:
                        frames.append(animated_frame)
                        frame_count += 1
                        
                # Add delay based on word count for natural timing
                words = text.split()
                delay = len(words) * 0.1  # 100ms per word
                if time.time() - start_time > delay:
                    break
                    
            cap.release()
            
            if not frames:
                print("No frames were generated")
                return None
                
            # Save animation with high quality
            output_path = "animation.mp4"
            imageio.mimsave(
                output_path,
                frames,
                fps=30,
                quality=8,
                macro_block_size=None
            )
            
            print(f"Saved animation to: {output_path}")
            return output_path
            
        except Exception as e:
            print(f"Error in animate_response: {e}")
            import traceback
            traceback.print_exc()
            return None

def create_interface():
    """Create the Gradio interface."""
    # Initialize the avatar
    try:
        avatar = ChatAvatar("assets/source_image.jpg")
    except Exception as e:
        print(f"Error initializing avatar: {e}")
        return None
        
    def chat(message, history):
        """Handle chat messages and generate responses."""
        try:
            # Generate response
            response = avatar.generate_response(message)
            
            # Generate speech and animation in parallel
            audio_file = avatar.text_to_speech(response)
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
            import traceback
            traceback.print_exc()
            return {
                "response": "I apologize, but I encountered an error.",
                "audio": None,
                "animation": None
            }
    
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
            example = gr.Button("Try an Example")
            
        # Event handlers
        send.click(chat, [msg, chatbot], [chatbot, video, audio])
        clear.click(lambda: None, None, chatbot)
        example.click(
            lambda: "Hi! How are you today?",
            None,
            msg
        )
        
    return demo

if __name__ == "__main__":
    demo = create_interface()
    if demo:
        demo.launch(debug=True)
    else:
        print("Failed to initialize the interface")
