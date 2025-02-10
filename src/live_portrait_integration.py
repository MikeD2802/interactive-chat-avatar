import os
import sys
import torch
import numpy as np
from pathlib import Path

class LivePortraitIntegration:
    def __init__(self, weights_dir="pretrained_weights"):
        self.weights_dir = Path(weights_dir)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.setup_environment()
        
    def setup_environment(self):
        """Setup LivePortrait environment and dependencies"""
        # Add LivePortrait to Python path
        live_portrait_path = Path("./LivePortrait")
        if not live_portrait_path.exists():
            raise RuntimeError(
                "LivePortrait repository not found. Please clone it first:\n"
                "git clone https://github.com/KwaiVGI/LivePortrait"
            )
        
        sys.path.append(str(live_portrait_path))
        
        # Check for pretrained weights
        if not self.weights_dir.exists():
            raise RuntimeError(
                "Pretrained weights not found. Please download them using:\n"
                "huggingface-cli download KwaiVGI/LivePortrait --local-dir pretrained_weights"
            )
    
    def initialize_model(self):
        """Initialize the LivePortrait model"""
        try:
            from LivePortrait.src.animator import Animator
            
            self.model = Animator(
                checkpoint_path=str(self.weights_dir / "humans" / "model.pth"),
                config_path=str(self.weights_dir / "humans" / "config.yaml"),
                device=self.device
            )
            return True
        except Exception as e:
            print(f"Failed to initialize LivePortrait model: {str(e)}")
            return False
    
    def generate_animation(self, source_image, expression_params):
        """Generate avatar animation based on expression parameters
        
        Args:
            source_image: PIL Image or path to source image
            expression_params: Dict containing animation parameters
                {
                    'smile': float (0-1),
                    'intensity': float (0-1),
                    'pose': Dict of pose parameters
                }
        
        Returns:
            Numpy array of animation frames
        """
        if self.model is None:
            if not self.initialize_model():
                return None
        
        try:
            # Convert expression params to LivePortrait format
            lp_params = self._convert_expression_params(expression_params)
            
            # Generate animation frames
            frames = self.model.animate(
                source_image,
                params=lp_params,
                output_path=None  # Don't save to file
            )
            
            return frames
            
        except Exception as e:
            print(f"Animation generation failed: {str(e)}")
            return None
    
    def _convert_expression_params(self, params):
        """Convert our expression parameters to LivePortrait format"""
        # This will need to be adjusted based on LivePortrait's exact API
        lp_params = {
            'expression': {
                'smile_intensity': params.get('smile', 0) * params.get('intensity', 1),
            },
            'pose': params.get('pose', {})
        }
        return lp_params

def setup_live_portrait():
    """Helper function to set up LivePortrait integration"""
    return LivePortraitIntegration()