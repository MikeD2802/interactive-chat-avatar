import os
import sys
import torch
import tempfile
from pathlib import Path
from src.utils.download import download_model, download_model_from_url

class AvatarGenerator:
    def __init__(self, checkpoint_path=None):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.checkpoint_path = checkpoint_path or 'checkpoints'
        self.temp_dir = tempfile.mkdtemp()
        self.initialized = False
        self.default_config = {
            'face_model_size': 512,    # 256 or 512
            'exp_scale': 1.0,
            'pose_style': 1.0,
            'use_enhancer': True,
            'batch_size': 1,
            'size': 512,
            'still_mode': False,
            'use_ref_video': False,
            'use_full_body': False,    # Full body mode
            'preload': True,           # Preload models
            'use_safetensor': True,    # Use safetensor models
        }
        
    async def initialize(self):
        if self.initialized:
            return
            
        # Create checkpoint directory
        os.makedirs(self.checkpoint_path, exist_ok=True)
        
        # Download required models
        model_urls = {
            'sadtalker_512': 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/SadTalker_V0.0.2_512.safetensors',
            'sadtalker_256': 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/SadTalker_V0.0.2_256.safetensors',
            'mapping_00229': 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/mapping_00229-model.pth.tar',
            'mapping_00109': 'https://github.com/OpenTalker/SadTalker/releases/download/v0.0.2/mapping_00109-model.pth.tar',
        }
        
        for model_name, url in model_urls.items():
            model_path = os.path.join(self.checkpoint_path, os.path.basename(url))
            if not os.path.exists(model_path):
                await download_model_from_url(url, model_path)
        
        # Import SadTalker modules
        sys.path.append('src/SadTalker')
        from src.face3d.models.facerecon_model import FaceReconModel
        from src.generate_batch import SadTalker
        
        # Initialize SadTalker with enhanced options
        self.sad_talker = SadTalker(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            use_safetensor=self.default_config['use_safetensor'],
            face_model_size=self.default_config['face_model_size']
        )
        
        if self.default_config['preload']:
            await self.preload_models()
            
        self.initialized = True
        
    def prepare_config(self, config=None):
        """Merge provided config with defaults"""
        if config is None:
            return self.default_config
            
        merged_config = {**self.default_config, **config}
        
        # Validate face model size
        if merged_config['face_model_size'] not in [256, 512]:
            merged_config['face_model_size'] = 512
            
        return merged_config
        
    async def generate_talking_avatar(self, source_image, audio_path=None, text=None, config=None):
        """Generate a talking avatar video with enhanced features"""
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
            
        # Generate video with enhanced features
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
                use_ref_video=full_config['use_ref_video'],
                use_full_body=full_config['use_full_body'],
                face_model_size=full_config['face_model_size']
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
            
        # Preload face recognition and other models
        if hasattr(self.sad_talker, 'preload'):
            await self.sad_talker.preload()
            
    def get_available_sizes(self):
        """Return available face model sizes"""
        return [256, 512]
        
    def get_available_features(self):
        """Return available features and their descriptions"""
        return {
            'still_mode': 'Reduce head movement',
            'use_enhancer': 'Enhance face quality',
            'use_full_body': 'Generate full body animation',
            'use_ref_video': 'Use reference video for motion',
            'face_model_size': 'Size of face model (256 or 512)',
        }
                
    def __del__(self):
        """Cleanup temporary directory on object destruction"""
        if hasattr(self, 'temp_dir') and os.path.exists(self.temp_dir):
            import shutil
            shutil.rmtree(self.temp_dir)