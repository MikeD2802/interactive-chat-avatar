import gradio as gr
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.llms import Ollama
from gtts import gTTS
import os
import json

class ChatAvatar:
    def __init__(self):
        # Initialize Ollama LLM
        self.llm = Ollama(model="llama2")
        
        # Initialize conversation history
        self.history = []
        
        # Define avatar personality prompt
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", "You are a helpful and friendly AI assistant. Respond naturally and conversationally."),
            ("user", "{input}")
        ])
    
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

def create_interface():
    avatar = ChatAvatar()
    
    def chat(message, history):
        # Generate response
        response = avatar.generate_response(message)
        
        # Convert to speech
        audio_file = avatar.text_to_speech(response)
        
        # Here you would trigger the LivePortrait animation
        # This is a placeholder for the animation logic
        animate_avatar(response)
        
        return response
    
    def animate_avatar(text):
        # Placeholder for LivePortrait integration
        # This would analyze the text sentiment and animate accordingly
        pass
    
    # Create Gradio interface
    iface = gr.ChatInterface(
        chat,
        title="AI Chat Avatar",
        description="Chat with an AI-powered avatar",
        examples=["Hello! How are you?", "Tell me about yourself"],
        theme="default"
    )
    
    return iface

if __name__ == "__main__":
    interface = create_interface()
    interface.launch()