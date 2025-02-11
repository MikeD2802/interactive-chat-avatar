import os
import sys
import logging
from dotenv import load_dotenv
load_dotenv()

from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM
import gradio as gr
from pathlib import Path
from PIL import Image
import numpy as np
from transformers import pipeline
import imageio  # make sure this import is at the top

from avatar_animation import AvatarAnimator
from live_portrait_integration import setup_live_portrait
from src.elevenlabs_tts import elevenlabs_text_to_speech

# Configure logging
logging.basicConfig(level=logging.DEBUG, 
                    format='%(asctime)s - %(levelname)s - %(message)s',
                    handlers=[
                        logging.StreamHandler(sys.stdout),
                        logging.FileHandler('avatar_debug.log')
                    ])

class ChatAvatar:
    def __init__(self, source_image_path):
        # Extensive logging for initialization
        logging.info(f"Initializing ChatAvatar with source image: {source_image_path}")
        
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
        
        # Verify and load source image with extensive error handling
        try:
            # Check image path exists
            source_path = Path(source_image_path)
            if not source_path.exists():
                logging.error(f"Source image not found at {source_image_path}")
                logging.error(f"Current working directory: {os.getcwd()}")
                logging.error(f"Contents of assets directory: {os.listdir('assets') if os.path.exists('assets') else 'assets directory not found'}")
                raise FileNotFoundError(f"Source image not found at {source_image_path}")
            
            # Open and validate image
            self.source_image = Image.open(source_path)
            logging.info(f"Image loaded successfully. Size: {self.source_image.size}, Mode: {self.source_image.mode}")
            
            # Additional image validation
            np_image = np.array(self.source_image)
            logging.info(f"Numpy image shape: {np_image.shape}, dtype: {np_image.dtype}")
        except Exception as e:
            logging.error(f"Error loading source image: {e}")
            raise
        
        # Setup LivePortrait and animator
        try:
            self.portrait = setup_live_portrait()
            self.animator = AvatarAnimator()
            logging.info("LivePortrait and Animator initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize LivePortrait or Animator: {e}")
            raise
    
    def generate_response(self, user_input):
        logging.info(f"Generating response for input: {user_input}")
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
        logging.info(f"Converting text to speech: {text[:50]}...")
        try:
            # Try ElevenLabs first
            audio_file = elevenlabs_text_to_speech(text)
            if audio_file:
                logging.info(f"ElevenLabs audio generated: {audio_file}")
                return audio_file
            
            # Fallback to gTTS
            from gtts import gTTS
            tts = gTTS(text=text, lang='en')
            tts.save("response.mp3")
            logging.info("Fallback to gTTS: response.mp3 created")
            return "response.mp3"
        except Exception as e:
            logging.error(f"TTS conversion error: {e}")
            # Fallback to gTTS
            from gtts import gTTS
            tts = gTTS(text=text, lang='en')
            tts.save("response.mp3")
            return "response.mp3"
    
    def animate_response(self, text):
        logging.info(f"Attempting to animate response for text: {text[:50]}...")
        
        # Analyze sentiment and generate expression parameters
        sentiment = self.sentiment_analyzer(text)
        expression = self.animator.generate_expression(sentiment[0])
        logging.info(f"Generated expression: {expression}")
        
        # Call LivePortrait integration to generate frames
        try:
            frames = self.portrait.generate_animation(self.source_image, expression_params=expression)
            
            # Extensive frame generation debugging
            if frames is None:
                logging.error("generate_animation returned None!")
                return None
            
            if not isinstance(frames, list):
                logging.error(f"generate_animation did not return a list. Got: {type(frames)}")
                return None
            
            logging.info(f"Number of frames returned: {len(frames)}")
            
            if len(frames) == 0:
                logging.error("No frames were generated. Check your LivePortrait integration!")
                return None
            
            # Convert the list of frames into a video file (animation.mp4)
            video_path = "animation.mp4"
            try:
                imageio.mimsave(video_path, frames, fps=30)
                logging.info(f"Video saved to {video_path}")
                return video_path
            except Exception as e:
                logging.error(f"Error creating video: {e}")
                return None
        
        except Exception as e:
            logging.error(f"Animation generation failed: {e}")
            return None

def create_interface(source_image_path):
    try:
        avatar = ChatAvatar(source_image_path)
        
        def chat(message, history):
            # Generate text response
            response = avatar.generate_response(message)
            
            # Generate speech
            audio_file = avatar.text_to_speech(response)
            
            # Generate animation
            animation = avatar.animate_response(response)
            
            # Logging for debugging
            logging.info(f"Chat response: {response}")
            logging.info(f"Audio file: {audio_file}")
            logging.info(f"Animation file: {animation}")
            
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
    except Exception as e:
        logging.error(f"Failed to create interface: {e}")
        raise

if __name__ == "__main__":
    # Default source image path
    source_image = "assets/source_image.jpg"
    
    # Create assets directory if it doesn't exist
    Path("assets").mkdir(exist_ok=True)
    
    # Check if source image exists
    if not Path(source_image).exists():
        logging.error(f"Please place a source image at {source_image}")
        print(f"Please place a source image at {source_image}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Contents of current directory: {os.listdir('.')}")
        exit(1)
    
    interface = create_interface(source_image)
    interface.launch(debug=True)  # Added debug mode for more verbose output