import os
import sys
import torch
import tempfile
from pathlib import Path
from src.utils.download import download_model

class AvatarGenerator:
    def __init__(self, checkpoint_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path = checkpoint_path or 'checkpoints'
        self.temp_dir = tempfile.mkdtemp()
        self.initialized = False
        
    async def initialize(self):
        if self.initialized:
            return
            
        # Download models if they don't exist
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path, exist_ok=True)
            await download_model('sadtalker')
            
        # Import SadTalker modules here to avoid loading them before model download
        sys.path.append('src/SadTalker')
        from src.face3d.models.facerecon_model import FaceReconModel
        from src.generate_batch import SadTalker
        
        self.sad_talker = SadTalker(
            checkpoint_path=self.checkpoint_path,
            device=self.device
        )
        self.initialized = True
        
    async def generate_talking_avatar(self, source_image, audio_path=None, text=None):
        """Generate a talking avatar video from either audio or text"""
        if not self.initialized:
            await self.initialize()
            
        # Create temporary paths for processing
        temp_source = os.path.join(self.temp_dir, 'source.jpg')
        temp_output = os.path.join(self.temp_dir, 'output.mp4')
        
        # Save source image if it's in memory
        if isinstance(source_image, bytes):
            with open(temp_source, 'wb') as f:
                f.write(source_image)
            source_path = temp_source
        else:
            source_path = source_image
            
        # Generate video
        try:
            result = await self.sad_talker.animate(
                source_path=source_path,
                audio_path=audio_path,
                text=text,
                output_path=temp_output,
                still_mode=False,
                use_enhancer=True,
                batch_size=1
            )
            
            # Read the generated video
            with open(temp_output, 'rb') as f:
                video_data = f.read()
                
            return video_data
            
        except Exception as e:
            print(f"Error generating avatar: {str(e)}")
            raise
            
        finally:
            # Cleanup temporary files
            if os.path.exists(temp_source):
                os.remove(temp_source)
            if os.path.exists(temp_output):
                os.remove(temp_output)
                
    def __del__(self):
        """Cleanup temporary directory on object destruction"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)