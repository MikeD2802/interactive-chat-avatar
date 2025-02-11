import cv2
import numpy as np
import torch
from PIL import Image
import mediapipe as mp

class RealTimeAnimator:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
    def process_frame(self, frame):
        """Process a single frame and extract facial landmarks."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            return self._extract_features(landmarks, frame.shape)
        return None
        
    def _extract_features(self, landmarks, frame_shape):
        """Extract relevant facial features from landmarks."""
        height, width = frame_shape[:2]
        points = []
        
        for landmark in landmarks.landmark:
            x = int(landmark.x * width)
            y = int(landmark.y * height)
            points.append((x, y))
            
        return np.array(points)
        
    def animate_frame(self, source_image, target_landmarks):
        """Animate the source image using target landmarks."""
        try:
            if isinstance(source_image, str):
                source_image = Image.open(source_image)
            source_array = np.array(source_image)
            
            # Extract source landmarks
            source_results = self.face_mesh.process(cv2.cvtColor(source_array, cv2.COLOR_BGR2RGB))
            if not source_results.multi_face_landmarks:
                print("No face detected in source image")
                return None
                
            source_landmarks = self._extract_features(source_results.multi_face_landmarks[0], source_array.shape)
            
            # Perform warping based on landmarks
            animated_frame = self._warp_image(source_array, source_landmarks, target_landmarks)
            return animated_frame
            
        except Exception as e:
            print(f"Error animating frame: {e}")
            return None
            
    def _warp_image(self, source_image, source_points, target_points):
        """Warp the source image to match target points."""
        if len(source_points) != len(target_points):
            return source_image
            
        transform = cv2.estimateAffinePartial2D(source_points, target_points)[0]
        if transform is None:
            return source_image
            
        return cv2.warpAffine(source_image, transform, (source_image.shape[1], source_image.shape[0]))
