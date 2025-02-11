import sys
import os
import logging

# Add LivePortrait directory to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
liveportrait_path = os.path.join(project_root, 'LivePortrait')
sys.path.insert(0, liveportrait_path)

try:
    from portrait.live_portrait import LivePortrait
    import torch
    import numpy as np
    from PIL import Image

    def setup_live_portrait():
        """
        Initialize and setup LivePortrait model
        
        Returns:
            LivePortrait instance or None if initialization fails
        """
        try:
            # Ensure CUDA or MPS is available
            device = torch.device('cuda' if torch.cuda.is_available() else 
                                  'mps' if torch.backends.mps.is_available() else 
                                  'cpu')
            
            # Initialize LivePortrait model
            model = LivePortrait(device=device)
            
            logging.info(f"LivePortrait initialized on {device}")
            return model
        
        except Exception as e:
            logging.error(f"Failed to initialize LivePortrait: {e}")
            return None

    def generate_avatar_animation(model, source_image, expression_params=None):
        """
        Generate avatar animation
        
        Args:
            model (LivePortrait): Initialized LivePortrait model
            source_image (PIL.Image): Source image for animation
            expression_params (dict, optional): Parameters for facial expression
        
        Returns:
            numpy.ndarray or None: Animated frames
        """
        if model is None:
            logging.error("LivePortrait model is not initialized")
            return None
        
        try:
            # If no expression params provided, use neutral expression
            if expression_params is None:
                expression_params = {'neutral': True}
            
            # Generate animation
            animated_frames = model.generate(source_image, **expression_params)
            
            return animated_frames
        
        except Exception as e:
            logging.error(f"Failed to generate avatar animation: {e}")
            return None

except ImportError as e:
    logging.error(f"Could not import LivePortrait: {e}")
    
    def setup_live_portrait():
        logging.error("LivePortrait is not available")
        return None
    
    def generate_avatar_animation(*args, **kwargs):
        logging.error("LivePortrait is not available")
        return None
