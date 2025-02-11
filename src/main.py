def create_interface():
    """Create enhanced Gradio interface."""
    try:
        avatar = ChatAvatar("assets/source_image.jpg")
    except Exception as e:
        print(f"Error initializing avatar: {e}")
        return None
    
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        gr.Markdown("# Enhanced Interactive Chat Avatar")
        
        with gr.Row():
            # Left column for avatar and controls
            with gr.Column(scale=1):
                video = gr.Video(label="Avatar")
                audio = gr.Audio(label="Response Audio")
                
                with gr.Accordion("Expression Controls", open=False):
                    eyes_weight = gr.Slider(0, 2, value=1.0, label="Eyes Expression")
                    mouth_weight = gr.Slider(0, 2, value=1.0, label="Mouth Expression")
                    eyebrows_weight = gr.Slider(0, 2, value=0.9, label="Eyebrows Expression")
                    nose_weight = gr.Slider(0, 2, value=0.8, label="Nose Movement")
                    
                with gr.Accordion("Animation Controls", open=False):
                    rotation = gr.Slider(-30, 30, value=0, label="Head Rotation")
                    scale = gr.Slider(0.5, 1.5, value=1.0, label="Scale")
                    fps = gr.Slider(15, 60, value=30, step=1, label="FPS")
                    quality = gr.Slider(1, 10, value=8, step=1, label="Video Quality")
                    
                with gr.Accordion("Advanced Settings", open=False):
                    smooth_factor = gr.Slider(0, 1, value=0.5, label="Motion Smoothing")
                    transition_frames = gr.Slider(5, 30, value=10, step=1, label="Transition Frames")
                    enable_retargeting = gr.Checkbox(value=True, label="Enable Retargeting")
                    stabilization = gr.Checkbox(value=True, label="Enable Stabilization")
                    
                with gr.Accordion("Voice Settings", open=False):
                    voice_selector = gr.Dropdown(
                        choices=["Rachel", "Bella", "Antoni", "Elli", "Josh", "Arnold", "Adam", "Sam"],
                        value="Rachel",
                        label="Voice"
                    )
                    voice_speed = gr.Slider(0.5, 2.0, value=1.0, label="Speech Speed")
                    voice_stability = gr.Slider(0, 1, value=0.5, label="Voice Stability")
            
            # Right column for chat
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(type="messages", height=400)
                with gr.Row():
                    msg = gr.Textbox(
                        show_label=False,
                        placeholder="Type your message...",
                        scale=8
                    )
                    send = gr.Button("Send", scale=1)
                
                with gr.Row():
                    clear = gr.Button("Clear Chat")
                    example = gr.Button("Try Example")
                    save_settings = gr.Button("Save Settings")
                
                with gr.Accordion("Debug Information", open=False):
                    debug_info = gr.JSON(label="Last Response Debug Info")
        
        def chat(message, history):
            """Enhanced chat function with parameter updates."""
            try:
                # Update animation parameters
                expression_params = {
                    'eyes': eyes_weight.value,
                    'mouth': mouth_weight.value,
                    'eyebrows': eyebrows_weight.value,
                    'nose': nose_weight.value
                }
                
                retarget_params = {
                    'enable': enable_retargeting.value,
                    'rotation': rotation.value,
                    'scale': scale.value,
                    'stabilization': stabilization.value
                }
                
                # Update animation settings
                avatar.animation_params.update({
                    'fps': fps.value,
                    'quality': quality.value,
                    'transition_frames': transition_frames.value,
                    'smooth_factor': smooth_factor.value
                })
                
                # Generate response and animations
                start_time = time.time()
                response = avatar.generate_response(message)
                
                # Update TTS settings
                avatar.tts_params = {
                    'voice': voice_selector.value,
                    'speed': voice_speed.value,
                    'stability': voice_stability.value
                }
                
                animation = avatar.animate_response(
                    response,
                    expression_params=expression_params,
                    retarget_params=retarget_params
                )
                audio = avatar.text_to_speech(response)
                
                # Prepare debug information
                debug_data = {
                    'processing_time': f"{time.time() - start_time:.2f}s",
                    'expression_params': expression_params,
                    'retarget_params': retarget_params,
                    'animation_params': avatar.animation_params,
                    'tts_params': avatar.tts_params,
                    'response_length': len(response),
                    'animation_path': animation,
                    'audio_path': audio
                }
                
                # Update debug info
                debug_info.update(debug_data)
                
                # Update chat history
                history.append({"role": "user", "content": message})
                history.append({"role": "assistant", "content": response})
                
                return history, animation, audio
                
            except Exception as e:
                print(f"Error in chat function: {e}")
                import traceback
                traceback.print_exc()
                return history, None, None
        
        # Event handlers
        msg.submit(chat, [msg, chatbot], [chatbot, video, audio])
        send.click(chat, [msg, chatbot], [chatbot, video, audio])
        clear.click(lambda: [], None, chatbot)
        example.click(
            lambda: "Hi! How are you today?",
            None,
            msg
        )
        
        def save_current_settings():
            """Save current settings to a JSON file."""
            settings = {
                'expression': {
                    'eyes': eyes_weight.value,
                    'mouth': mouth_weight.value,
                    'eyebrows': eyebrows_weight.value,
                    'nose': nose_weight.value
                },
                'animation': {
                    'fps': fps.value,
                    'quality': quality.value,
                    'transition_frames': transition_frames.value,
                    'smooth_factor': smooth_factor.value
                },
                'retargeting': {
                    'enable': enable_retargeting.value,
                    'rotation': rotation.value,
                    'scale': scale.value,
                    'stabilization': stabilization.value
                },
                'voice': {
                    'name': voice_selector.value,
                    'speed': voice_speed.value,
                    'stability': voice_stability.value
                }
            }
            
            try:
                os.makedirs('config', exist_ok=True)
                with open('config/avatar_settings.json', 'w') as f:
                    json.dump(settings, f, indent=2)
                return "Settings saved successfully!"
            except Exception as e:
                return f"Error saving settings: {e}"
        
        save_settings.click(save_current_settings, None, None)
    
    return demo

if __name__ == "__main__":
    demo = create_interface()
    if demo:
        demo.launch(debug=True)
    else:
        print("Failed to initialize the interface")