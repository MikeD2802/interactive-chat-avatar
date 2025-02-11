import os
import gradio as gr
import asyncio
import tempfile
from pathlib import Path
from avatar_generator import AvatarGenerator
from elevenlabs import generate, save, voices, Voice
import ollama

class EnhancedChatAvatar:
    def __init__(self):
        self.avatar_generator = AvatarGenerator()
        self.conversation_history = []
        self.temp_dir = tempfile.mkdtemp()
        self.elevenlabs_voice_id = None
        self.voice_list = None
        
    async def initialize(self):
        """Initialize all required components"""
        await self.avatar_generator.initialize()
        self.llm = ollama.Client()
        await self.llm.pull('llama2')
        
    async def get_elevenlabs_voices(self, api_key):
        """Fetch available voices from ElevenLabs"""
        try:
            os.environ["ELEVEN_API_KEY"] = api_key
            available_voices = voices()
            voice_names = [(voice.voice_id, voice.name) for voice in available_voices]
            return voice_names, "✓ API Key valid. Voices loaded."
        except Exception as e:
            return [], f"✗ Error: {str(e)}"
            
    async def generate_response(self, 
                              message, 
                              history, 
                              voice_id, 
                              expression_intensity,
                              head_movement,
                              enhance_quality):
        """Generate LLM response and avatar animation with customizations"""
        try:
            # Get LLM response
            response = await self.llm.generate('llama2', prompt=message)
            response_text = response['response']
            
            # Generate speech using selected ElevenLabs voice
            audio_path = os.path.join(self.temp_dir, 'response.wav')
            audio = generate(
                text=response_text,
                voice=voice_id,
                model="eleven_monolingual_v1"
            )
            save(audio, audio_path)
            
            # Generate talking avatar with custom parameters
            source_image = "assets/source_image.jpg"  # Default avatar image
            video_data = await self.avatar_generator.generate_talking_avatar(
                source_image=source_image,
                audio_path=audio_path,
                config={
                    'exp_scale': float(expression_intensity),
                    'pose_style': float(head_movement),
                    'use_enhancer': enhance_quality
                }
            )
            
            # Save video temporarily
            video_path = os.path.join(self.temp_dir, 'response.mp4')
            with open(video_path, 'wb') as f:
                f.write(video_data)
                
            # Update conversation history
            history.append((message, response_text))
            
            return history, video_path, "Generation completed successfully!"
            
        except Exception as e:
            error_msg = f"Error generating response: {str(e)}"
            print(error_msg)
            return history, None, error_msg
            
    def create_interface(self):
        """Create enhanced Gradio interface"""
        with gr.Blocks(theme=gr.themes.Soft()) as demo:
            # Header
            with gr.Row():
                gr.Markdown("# 🤖 Enhanced AI Avatar Chat")
                
            # API Key Setup
            with gr.Row():
                with gr.Column():
                    api_key_input = gr.Textbox(
                        label="ElevenLabs API Key",
                        type="password",
                        placeholder="Enter your API key here..."
                    )
                    voice_dropdown = gr.Dropdown(
                        label="Select Voice",
                        choices=[],
                        interactive=True
                    )
                    api_status = gr.Markdown("Enter API key to load voices")
                    
            # Main Interface
            with gr.Row():
                # Left Column - Chat Interface
                with gr.Column(scale=2):
                    chatbot = gr.Chatbot(
                        height=500,
                        show_label=False,
                        container=True,
                        bubble_full_width=False
                    )
                    with gr.Row():
                        msg = gr.Textbox(
                            placeholder="Type your message here...",
                            scale=8,
                            container=False
                        )
                        submit_btn = gr.Button(
                            "Send",
                            scale=1,
                            variant="primary"
                        )
                        
                # Right Column - Avatar and Controls
                with gr.Column(scale=3):
                    video = gr.Video(label="Avatar Animation")
                    with gr.Row():
                        expression_slider = gr.Slider(
                            minimum=0.5,
                            maximum=1.5,
                            value=1.0,
                            step=0.1,
                            label="Expression Intensity"
                        )
                        head_movement_slider = gr.Slider(
                            minimum=0,
                            maximum=2,
                            value=1,
                            step=0.1,
                            label="Head Movement"
                        )
                    with gr.Row():
                        enhance_checkbox = gr.Checkbox(
                            label="Enhance Quality",
                            value=True,
                            info="Uses GFPGAN to improve output quality"
                        )
                    status_text = gr.Markdown("System ready")
                    
            # Footer Controls
            with gr.Row():
                clear_btn = gr.Button("Clear Chat", variant="secondary")
                example_btn = gr.Button("Load Example", variant="secondary")
                
            # Event handlers
            async def update_voices(api_key):
                voices, message = await self.get_elevenlabs_voices(api_key)
                return {
                    voice_dropdown: gr.Dropdown(choices=voices),
                    api_status: message
                }
                
            api_key_input.change(
                update_voices,
                inputs=[api_key_input],
                outputs=[voice_dropdown, api_status]
            )
            
            async def process_message(message, history, voice_id, exp, head, enhance):
                if not message.strip():
                    return history, None, "Please enter a message"
                    
                return await self.generate_response(
                    message, history, voice_id,
                    exp, head, enhance
                )
                
            submit_btn.click(
                process_message,
                inputs=[
                    msg, chatbot, voice_dropdown,
                    expression_slider, head_movement_slider,
                    enhance_checkbox
                ],
                outputs=[chatbot, video, status_text],
                api_name="chat"
            )
            
            clear_btn.click(
                lambda: ([], None, "Chat cleared"),
                outputs=[chatbot, video, status_text]
            )
            
            example_btn.click(
                lambda: (
                    [["Hello!", "Hi! How can I help you today?"]],
                    None,
                    "Example loaded"
                ),
                outputs=[chatbot, video, status_text]
            )
            
        return demo
        
    def launch(self):
        """Launch the Gradio interface"""
        demo = self.create_interface()
        
        async def start_app():
            await self.initialize()
            demo.launch(share=True)
            
        asyncio.run(start_app())
        
    def cleanup(self):
        """Cleanup temporary files"""
        if os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)
            
if __name__ == "__main__":
    chat_avatar = EnhancedChatAvatar()
    try:
        chat_avatar.launch()
    finally:
        chat_avatar.cleanup()