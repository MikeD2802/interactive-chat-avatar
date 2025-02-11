from transformers import pipeline
import numpy as np

class AvatarAnimator:
    def __init__(self, sentiment_analyzer=None):
        # Use provided sentiment analyzer or create a default one
        self.sentiment_analyzer = sentiment_analyzer or pipeline(
            "sentiment-analysis", 
            model="distilbert-base-uncased-finetuned-sst-2-english"
        )
    
    def generate_expression(self, sentiment):
        """
        Generate expression parameters based on sentiment analysis
        
        Args:
            sentiment (dict): Sentiment analysis result from Hugging Face pipeline
        
        Returns:
            dict: Expression parameters for animation
        """
        # Map sentiment to expression parameters
        if sentiment['label'] == 'POSITIVE':
            # More positive sentiment leads to more intense smile
            smile_intensity = sentiment['score'] * 0.8  # Scale smile intensity
            return {
                'smile': True,
                'intensity': smile_intensity,
                'pose': {
                    'head_pitch': np.random.uniform(-0.1, 0.1),  # Slight head tilt
                    'head_yaw': np.random.uniform(-0.05, 0.05)   # Subtle head turn
                }
            }
        else:
            # Neutral or negative sentiment
            return {
                'smile': False,
                'intensity': 0.2,  # Minimal expression
                'pose': {
                    'head_pitch': np.random.uniform(-0.05, 0.05),
                    'head_yaw': 0
                }
            }
