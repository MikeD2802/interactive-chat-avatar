import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import torch
import math
from expression_controller import ExpressionController
import time

class FaceProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize expression controller
        self.expression_controller = ExpressionController()
        
        # Initialize smooth buffers for landmark stabilization
        self.smooth_factor = 0.5
        self.previous_landmarks = None
        
        # Enhanced facial feature indices from LivePortrait
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.EYEBROWS = [70, 63, 105, 66, 107, 336, 296, 334, 293, 300]
        self.NOSE = [1, 2, 98, 327, 331, 297]
        
        # Advanced parameters from LivePortrait
        self.expression_params = {
            'blend_factor': 0.7,
            'transition_speed': 0.3,
            'mouth_emphasis': 1.2,
            'eye_emphasis': 1.1,
            'smooth_factor': 0.5
        }
        
        # Initialize retargeting control
        self.retarget_controller = {
            'enable': True,
            'scale': 1.0,
            'rotation': 0.0,
            'translation': np.zeros(2)
        }
        
        # Initialize region weights for fine control
        self.region_weights = {
            'eyes': 1.0,
            'mouth': 1.0,
            'nose': 0.8,
            'eyebrows': 0.9,
            'face_outline': 0.7
        }
        
    def process_frame(self, frame, sentiment=None):
        """Process a single frame with enhanced expressions."""
        try:
            # Detect face landmarks
            points = self.detect_face(frame)
            if points is None:
                return None
            
            # Apply advanced stabilization
            points = self._apply_stabilization(points)
            
            # Apply retargeting control
            if self.retarget_controller['enable']:
                points = self._apply_retargeting(points)
            
            # Update expression controller with regional weights
            expression_state = self.expression_controller.update(
                time.time(),
                sentiment=sentiment,
                region_weights=self.region_weights
            )
            
            # Generate enhanced face mask
            mask = self._generate_enhanced_mask(frame, points)
            
            return {
                'landmarks': points,
                'mask': mask,
                'face_detected': True,
                'expression_state': expression_state
            }
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None
            
    def _apply_stabilization(self, points):
        """Apply advanced landmark stabilization."""
        if self.previous_landmarks is None:
            self.previous_landmarks = points
            return points
            
        # Calculate velocity for each point
        velocity = points - self.previous_landmarks
        
        # Apply adaptive smoothing based on velocity
        smooth_factor = np.clip(
            self.smooth_factor * (1.0 - np.linalg.norm(velocity, axis=1) / 100.0),
            0.1,
            0.9
        )
        
        # Apply smoothing with variable factor
        smoothed_points = np.zeros_like(points)
        for i in range(len(points)):
            smoothed_points[i] = self.previous_landmarks[i] + \
                               smooth_factor[i] * (points[i] - self.previous_landmarks[i])
        
        self.previous_landmarks = smoothed_points
        return smoothed_points
        
    def _apply_retargeting(self, points):
        """Apply retargeting control to landmarks."""
        # Apply scale
        center = np.mean(points, axis=0)
        scaled_points = center + (points - center) * self.retarget_controller['scale']
        
        # Apply rotation
        angle = np.radians(self.retarget_controller['rotation'])
        rot_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        rotated_points = np.dot(scaled_points - center, rot_matrix.T) + center
        
        # Apply translation
        final_points = rotated_points + self.retarget_controller['translation']
        
        return final_points
        
    def _generate_enhanced_mask(self, image, points):
        """Generate enhanced face mask with regional control."""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        
        # Draw face regions with different weights
        for region, weight in self.region_weights.items():
            if region == 'face_outline':
                pts = points[self.FACE_OVAL]
            elif region == 'eyes':
                pts = np.vstack([points[self.LEFT_EYE], points[self.RIGHT_EYE]])
            elif region == 'mouth':
                pts = points[self.LIPS]
            elif region == 'nose':
                pts = points[self.NOSE]
            elif region == 'eyebrows':
                pts = points[self.EYEBROWS]
            
            cv2.fillPoly(mask, [pts.astype(int)], int(255 * weight))
        
        # Apply advanced feathering
        mask = cv2.GaussianBlur(mask, (31, 31), 11)
        return mask
        
    def set_region_weights(self, weights):
        """Update region weights for fine control."""
        self.region_weights.update(weights)
        
    def set_retarget_params(self, params):
        """Update retargeting parameters."""
        self.retarget_controller.update(params)
        
    def set_expression_params(self, params):
        """Update expression parameters."""
        self.expression_params.update(params)