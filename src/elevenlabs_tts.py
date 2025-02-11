import os
from elevenlabs_mcp.elevenlabs_api import generate_audio_simple

def elevenlabs_text_to_speech(text, voice_id=None):
    """
    Generate audio from text using ElevenLabs MCP server integration.
    
    Parameters:
        text (str): The text to convert.
        voice_id (str, optional): If provided, uses the specified voice. Otherwise, defaults are used.
    
    Returns:
        str: Path to the generated audio file.
    """
    try:
        # Use environment variables for configuration
        api_key = os.environ.get('ELEVENLABS_API_KEY')
        default_voice_id = os.environ.get('ELEVENLABS_VOICE_ID', voice_id)
        
        if not api_key:
            print("ERROR: ELEVENLABS_API_KEY not set in environment variables.")
            return None
        
        # Call the ElevenLabs MCP function
        audio_file = generate_audio_simple(
            text, 
            api_key=api_key, 
            voice_id=default_voice_id
        )
        return audio_file
    except Exception as e:
        print("ElevenLabs TTS error:", e)
        return None

if __name__ == "__main__":
    sample_text = "Hello from ElevenLabs TTS integration!"
    result = elevenlabs_text_to_speech(sample_text)
    print("Audio file generated:", result)