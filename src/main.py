import os
import gradio as gr
import asyncio
import tempfile
from pathlib import Path
from avatar_generator import AvatarGenerator
from elevenlabs import generate, save
import ollama

class ChatAvatar:
    def __init__(self):
        self.avatar_generator = AvatarGenerator()
        self.conversation_history = []
        self.temp_dir = tempfile.mkdtemp()
        
    async def initialize(self):
        """Initialize all required components"""
        await self.avatar_generator.initialize()
        self.llm = ollama.Client()
        await self.llm.pull('llama2')
        
    async def generate_response(self, message, history):
        """Generate LLM response and avatar animation"""
        try:
            # Get LLM response
            response = await self.llm.generate('llama2', prompt=message)
            response_text = response['response']
            
            # Generate speech using ElevenLabs
            audio_path = os.path.join(self.temp_dir, 'response.wav')
            audio = generate(
                text=response_text,
                voice="Rachel",
                model="eleven_monolingual_v1"
            )
            save(audio, audio_path)
            
            # Generate talking avatar
            source_image = "assets/source_image.jpg"  # Your default avatar image
            video_data = await self.avatar_generator.generate_talking_avatar(
                source_image=source_image,
                audio_path=audio_path
            )
            
            # Save video temporarily
            video_path = os.path.join(self.temp_dir, 'response.mp4')
            with open(video_path, 'wb') as f:
                f.write(video_data)
                
            # Update conversation history
            history.append((message, response_text))
            
            return history, video_path
            
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            return history, None
            
    def create_interface(self):
        """Create Gradio interface"""
        with gr.Blocks() as demo:
            gr.Markdown("# Interactive Chat Avatar")
            
            with gr.Row():
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot()
                    msg = gr.Textbox(placeholder="Type your message here...")
                    clear = gr.Button("Clear")
                    
                with gr.Column(scale=3):
                    video = gr.Video(label="Avatar")
                    
            async def user_message(message, history):
                if message:
                    history, video_path = await self.generate_response(message, history)
                    return "", history, video_path
                return "", history, None
                
            msg.submit(user_message, [msg, chatbot], [msg, chatbot, video])
            clear.click(lambda: ([], None), outputs=[chatbot, video])
            
        return demo
        
    def launch(self):
        """Launch the Gradio interface"""
        demo = self.create_interface()
        
        async def start_app():
            await self.initialize()
            demo.launch()
            
        asyncio.run(start_app())
        
    def cleanup(self):
        """Cleanup temporary files"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            
if __name__ == "__main__":
    chat_avatar = ChatAvatar()
    try:
        chat_avatar.launch()
    finally:
        chat_avatar.cleanup()