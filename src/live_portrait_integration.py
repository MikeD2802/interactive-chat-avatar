import os
import sys
import logging
from pathlib import Path

try:
    import torch
    import numpy as np
    from PIL import Image
except ImportError:
    print("Please install required dependencies: torch, numpy, Pillow")
    sys.exit(1)

class LivePortraitIntegration:
    def __init__(self, weights_dir="pretrained_weights"):
        # Configure logging
        logging.basicConfig(level=logging.INFO, 
                            format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Set weights directory
        self.weights_dir = Path(weights_dir)
        
        # Determine device
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 
            'mps' if torch.backends.mps.is_available() else 
            'cpu'
        )
        
        # Model and related attributes
        self.model = None
        
        # Setup environment and initialize
        try:
            self.setup_environment()
            self.initialize_model()
        except Exception as e:
            logging.error(f"LivePortrait initialization failed: {e}")
            raise
    
    def setup_environment(self):
        """Setup LivePortrait environment and dependencies"""
        # Add LivePortrait to Python path
        live_portrait_path = Path("./LivePortrait")
        
        if not live_portrait_path.exists():
            logging.warning("LivePortrait repository not found. Attempting to clone...")
            import subprocess
            try:
                subprocess.run(["git", "clone", "https://github.com/KwaiVGI/LivePortrait"], 
                               check=True)
            except subprocess.CalledProcessError:
                logging.error("Failed to clone LivePortrait repository")
                raise RuntimeError("Could not clone LivePortrait repository")
        
        # Add to system path
        sys.path.append(str(live_portrait_path))
        
        # Check for pretrained weights
        if not self.weights_dir.exists() or not any(self.weights_dir.iterdir()):
            logging.warning("Pretrained weights not found. Please download them.")
            raise RuntimeError(
                "Pretrained weights not found. Download using:\n"
                "huggingface-cli download KwaiVGI/LivePortrait "
                "--local-dir pretrained_weights"
            )
    
    def initialize_model(self):
        """Initialize the LivePortrait model"""
        try:
            # Dynamically import to avoid early import errors
            from LivePortrait.src.portrait.live_portrait import LivePortrait
            
            # Path to the model configuration
            config_path = self.weights_dir / "humans" / "config.yaml"
            model_path = self.weights_dir / "humans" / "model.pth"
            
            # Validate paths
            if not config_path.exists() or not model_path.exists():
                raise FileNotFoundError(f"Model files not found in {self.weights_dir}")
            
            # Debug print for model paths
            print(f"Loading LivePortrait model from:")
            print(f"  Config: {config_path}")
            print(f"  Model: {model_path}")
            
            # Initialize model
            self.model = LivePortrait(
                checkpoint_path=str(model_path),
                config_path=str(config_path),
                device=self.device
            )
            
            print(f"LivePortrait model initialized successfully on {self.device}")
            return True
        except ImportError as e:
            logging.error(f"Failed to import LivePortrait: {e}")
            raise
        except Exception as e:
            logging.error(f"Model initialization failed: {e}")
            raise
    
    def generate_animation(self, source_image, expression_params):
        """Generate avatar animation based on expression parameters"""
        if self.model is None:
            logging.error("Model not initialized")
            return None
        
        try:
            # Debug print for function inputs
            print("Source image type:", type(source_image))
            print("Source image size:", source_image.size)
            print("Expression parameters:", expression_params)
            
            # Convert image to numpy if it's a PIL Image
            if isinstance(source_image, Image.Image):
                source_image = np.array(source_image)
            
            # Default expression if not provided
            if expression_params is None:
                expression_params = {
                    'smile': 0.5,
                    'intensity': 0.5
                }
            
            # Debug: show converted source image details
            print("Numpy image shape:", source_image.shape)
            print("Numpy image dtype:", source_image.dtype)
            
            # Convert expression params to LivePortrait format
            lp_params = self._convert_expression_params(expression_params)
            print("Converted LivePortrait parameters:", lp_params)
            
            # Generate animation frames
            frames = self.model.animate(
                source_image, 
                params=lp_params,
                output_path=None  # Don't save to file
            )
            
            # Debug frames
            if frames is not None:
                print("Generated frames:")
                print("  Number of frames:", len(frames))
                if len(frames) > 0:
                    print("  First frame shape:", frames[0].shape)
                    print("  First frame dtype:", frames[0].dtype)
            
            return frames
            
        except Exception as e:
            print(f"Animation generation failed: {str(e)}")
            import traceback
            traceback.print_exc()
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