import cv2
import numpy as np
import mediapipe as mp
from PIL import Image
import torch
import math

class FaceProcessor:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # Initialize smooth buffers for landmark stabilization
        self.smooth_factor = 0.5
        self.previous_landmarks = None
        
        # Landmark indices for different facial features
        self.FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
        self.LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # Gaze correction parameters
        self.gaze_correction = True
        self.gaze_threshold = 0.3
        
    def detect_face(self, image):
        """Detect facial landmarks in an image."""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_image)
        
        if not results.multi_face_landmarks:
            return None
            
        landmarks = results.multi_face_landmarks[0]
        points = self._landmarks_to_points(landmarks, image.shape)
        
        # Apply smoothing
        if self.previous_landmarks is not None:
            points = self._smooth_landmarks(points)
            
        self.previous_landmarks = points
        return points
        
    def _landmarks_to_points(self, landmarks, image_shape):
        """Convert landmarks to normalized points."""
        height, width = image_shape[:2]
        points = []
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))
            
        return np.array(points)
        
    def _smooth_landmarks(self, current_points):
        """Apply temporal smoothing to landmarks."""
        if self.previous_landmarks is None:
            return current_points
            
        return self.previous_landmarks + self.smooth_factor * (current_points - self.previous_landmarks)
        
    def correct_gaze(self, points):
        """Correct gaze direction based on eye landmarks."""
        if not self.gaze_correction:
            return points
            
        left_eye_center = np.mean(points[self.LEFT_EYE], axis=0)
        right_eye_center = np.mean(points[self.RIGHT_EYE], axis=0)
        
        # Calculate gaze direction
        gaze_direction = right_eye_center - left_eye_center
        gaze_angle = math.atan2(gaze_direction[1], gaze_direction[0])
        
        # Apply correction if gaze deviation is above threshold
        if abs(gaze_angle) > self.gaze_threshold:
            correction_matrix = cv2.getRotationMatrix2D(
                tuple(np.mean([left_eye_center, right_eye_center], axis=0)),
                math.degrees(gaze_angle),
                1.0
            )
            points = cv2.transform(points.reshape(1, -1, 2), correction_matrix).reshape(-1, 2)
            
        return points
        
    def get_face_mask(self, image, points):
        """Generate a face mask using landmarks."""
        mask = np.zeros(image.shape[:2], dtype=np.uint8)
        face_points = points[self.FACE_OVAL]
        cv2.fillPoly(mask, [face_points.astype(int)], 255)
        return mask
        
    def process_frame(self, frame):
        """Process a single frame."""
        try:
            # Detect face landmarks
            points = self.detect_face(frame)
            if points is None:
                return None
                
            # Apply gaze correction
            points = self.correct_gaze(points)
            
            # Generate face mask
            mask = self.get_face_mask(frame, points)
            
            return {
                'landmarks': points,
                'mask': mask,
                'face_detected': True
            }
            
        except Exception as e:
            print(f"Error processing frame: {e}")
            return None
            
    def animate_frame(self, source_image, target_points):
        """Animate source image using target points."""
        try:
            if isinstance(source_image, str):
                source_image = cv2.imread(source_image)
                
            source_points = self.detect_face(source_image)
            if source_points is None:
                return None
                
            # Calculate transformation
            transform = cv2.estimateAffinePartial2D(
                source_points.astype(np.float32),
                target_points.astype(np.float32)
            )[0]
            
            if transform is None:
                return source_image
                
            # Apply transformation
            animated = cv2.warpAffine(
                source_image,
                transform,
                (source_image.shape[1], source_image.shape[0]),
                borderMode=cv2.BORDER_REPLICATE
            )
            
            # Blend using face mask
            mask = self.get_face_mask(source_image, source_points)
            mask = cv2.warpAffine(
                mask,
                transform,
                (source_image.shape[1], source_image.shape[0])
            )
            
            mask = cv2.GaussianBlur(mask, (19, 19), 0)
            mask = mask.astype(float) / 255.0
            mask = np.expand_dims(mask, axis=-1)
            
            result = animated * mask + source_image * (1 - mask)
            return result.astype(np.uint8)
            
        except Exception as e:
            print(f"Error animating frame: {e}")
            return None
