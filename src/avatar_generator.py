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
        self.default_config = {
            'exp_scale': 1.0,
            'pose_style': 1.0,
            'use_enhancer': True,
            'batch_size': 1,
            'size': 256,
            'still_mode': False,
            'use_ref_video': False
        }
        
    async def initialize(self):
        if self.initialized:
            return
            
        # Download models if they don't exist
        if not os.path.exists(self.checkpoint_path):
            os.makedirs(self.checkpoint_path, exist_ok=True)
            await download_model('sadtalker')
            
        # Import SadTalker modules
        sys.path.append('src/SadTalker')
        from src.face3d.models.facerecon_model import FaceReconModel
        from src.generate_batch import SadTalker
        
        self.sad_talker = SadTalker(
            checkpoint_path=self.checkpoint_path,
            device=self.device
        )
        self.initialized = True
        
    def prepare_config(self, config=None):
        """Merge provided config with defaults"""
        if config is None:
            return self.default_config
            
        return {
            **self.default_config,
            **config
        }
        
    async def generate_talking_avatar(self, source_image, audio_path=None, text=None, config=None):
        """Generate a talking avatar video with customization options"""
        if not self.initialized:
            await self.initialize()
            
        # Prepare configuration
        full_config = self.prepare_config(config)
        
        # Create temporary paths
        temp_source = os.path.join(self.temp_dir, 'source.jpg')
        temp_output = os.path.join(self.temp_dir, 'output.mp4')
        
        # Save source image if it's in memory
        if isinstance(source_image, bytes):
            with open(temp_source, 'wb') as f:
                f.write(source_image)
            source_path = temp_source
        else:
            source_path = source_image
            
        # Generate video with customizations
        try:
            result = await self.sad_talker.animate(
                source_path=source_path,
                audio_path=audio_path,
                text=text,
                output_path=temp_output,
                still_mode=full_config['still_mode'],
                use_enhancer=full_config['use_enhancer'],
                batch_size=full_config['batch_size'],
                size=full_config['size'],
                pose_style=full_config['pose_style'],
                exp_scale=full_config['exp_scale'],
                use_ref_video=full_config['use_ref_video']
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
                
    async def preload_models(self):
        """Preload models for faster generation"""
        if not self.initialized:
            await self.initialize()
            
        # Preload face recognition model
        if hasattr(self.sad_talker, 'preload'):
            await self.sad_talker.preload()
                
    def __del__(self):
        """Cleanup temporary directory on object destruction"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)