import numpy as np
from dataclasses import dataclass
import math

@dataclass
class ExpressionConfig:
    blink_interval: float = 4.0  # Average time between blinks in seconds
    blink_duration: float = 0.2   # Duration of a blink in seconds
    head_movement_scale: float = 0.3  # Scale of random head movements
    expression_strength: float = 1.0  # Overall expression intensity
    lip_sync_scale: float = 1.0  # Scale of lip movement for speech

class ExpressionController:
    def __init__(self):
        self.config = ExpressionConfig()
        self.last_blink_time = 0
        self.is_blinking = False
        self.blink_progress = 0
        
        # Expression state
        self.current_expression = 'neutral'
        self.expression_blend = 0.0
        self.head_rotation = np.zeros(3)  # pitch, yaw, roll
        self.last_update_time = 0
        
    def update(self, timestamp, sentiment=None, is_speaking=False):
        """Update expression state based on time and context."""
        dt = timestamp - self.last_update_time
        self.last_update_time = timestamp
        
        # Update blink state
        self._update_blink(timestamp)
        
        # Update head movement
        self._update_head_movement(dt)
        
        # Update expression based on sentiment
        if sentiment:
            self._update_expression(sentiment, dt)
        
        # Get current state
        state = {
            'blink': self.is_blinking,
            'head_rotation': self.head_rotation.copy(),
            'expression': self.current_expression,
            'expression_blend': self.expression_blend,
            'is_speaking': is_speaking
        }
        
        return state
        
    def _update_blink(self, timestamp):
        """Update blinking state."""
        # Check if it's time for a new blink
        if not self.is_blinking and (timestamp - self.last_blink_time) > self.config.blink_interval:
            self.is_blinking = True
            self.blink_progress = 0
            self.last_blink_time = timestamp
            
        # Update blink progress
        if self.is_blinking:
            self.blink_progress += 0.1
            if self.blink_progress >= 1.0:
                self.is_blinking = False
                self.blink_progress = 0
                
    def _update_head_movement(self, dt):
        """Update random head movements."""
        # Add small random movements
        random_movement = np.random.normal(0, 0.1, 3) * self.config.head_movement_scale
        self.head_rotation += random_movement * dt
        
        # Apply damping
        self.head_rotation *= 0.95
        
        # Clamp rotation values
        self.head_rotation = np.clip(self.head_rotation, -0.5, 0.5)
        
    def _update_expression(self, sentiment, dt):
        """Update facial expression based on sentiment."""
        target_blend = 0.0
        
        if sentiment['label'] == 'POSITIVE':
            self.current_expression = 'happy'
            target_blend = sentiment['score'] * self.config.expression_strength
        elif sentiment['label'] == 'NEGATIVE':
            self.current_expression = 'sad'
            target_blend = sentiment['score'] * self.config.expression_strength
            
        # Smoothly blend to target expression
        blend_speed = 2.0 * dt
        self.expression_blend += (target_blend - self.expression_blend) * blend_speed
        
    def apply_expression(self, landmarks):
        """Apply current expression state to facial landmarks."""
        if landmarks is None:
            return None
            
        modified_landmarks = landmarks.copy()
        
        # Apply head rotation
        rotation_matrix = self._get_rotation_matrix()
        modified_landmarks = np.dot(modified_landmarks, rotation_matrix)
        
        # Apply expression deformation
        if self.current_expression != 'neutral':
            modified_landmarks = self._apply_expression_deformation(modified_landmarks)
            
        # Apply blink
        if self.is_blinking:
            modified_landmarks = self._apply_blink(modified_landmarks)
            
        return modified_landmarks
        
    def _get_rotation_matrix(self):
        """Get 3D rotation matrix from current head rotation."""
        pitch, yaw, roll = self.head_rotation
        
        # Create rotation matrices for each axis
        def rotation_x(angle):
            return np.array([
                [1, 0, 0],
                [0, math.cos(angle), -math.sin(angle)],
                [0, math.sin(angle), math.cos(angle)]
            ])
            
        def rotation_y(angle):
            return np.array([
                [math.cos(angle), 0, math.sin(angle)],
                [0, 1, 0],
                [-math.sin(angle), 0, math.cos(angle)]
            ])
            
        def rotation_z(angle):
            return np.array([
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1]
            ])
            
        # Combine rotations
        R = np.dot(rotation_z(roll), np.dot(rotation_y(yaw), rotation_x(pitch)))
        return R
        
    def _apply_expression_deformation(self, landmarks):
        """Apply expression-specific deformations to landmarks."""
        modified = landmarks.copy()
        
        if self.current_expression == 'happy':
            # Raise cheeks and corners of mouth
            mouth_indices = [48, 54]  # Example indices for mouth corners
            for idx in mouth_indices:
                modified[idx, 1] -= 5 * self.expression_blend
                
        elif self.current_expression == 'sad':
            # Lower corners of mouth and eyebrows
            mouth_indices = [48, 54]  # Example indices for mouth corners
            for idx in mouth_indices:
                modified[idx, 1] += 5 * self.expression_blend
                
        return modified
        
    def _apply_blink(self, landmarks):
        """Apply blink animation to eye landmarks."""
        modified = landmarks.copy()
        
        # Example eye landmark indices
        upper_eyelid = [159, 145, 33]  # Example indices
        lower_eyelid = [145, 159, 163]  # Example indices
        
        blink_amount = math.sin(self.blink_progress * math.pi) * 4
        
        # Move upper eyelids down and lower eyelids up
        for idx in upper_eyelid:
            modified[idx, 1] += blink_amount
        for idx in lower_eyelid:
            modified[idx, 1] -= blink_amount
            
        return modified