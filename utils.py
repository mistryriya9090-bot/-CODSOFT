import cv2
import numpy as np
from keras_facenet import FaceNet
import os

class FaceRecognitionUtils:
    def __init__(self):
        # Initialize FaceNet for embeddings
        self.facenet = FaceNet()
        
        # Initialize OpenCV Haar Cascade for face detection
        # Using the standard frontal face cascade
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

    def extract_face(self, img):
        """Detect and crop the first face from an image using Haar Cascade."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        
        if len(faces) > 0:
            # We take only the first face detected
            (x, y, w, h) = faces[0]
            
            # Pad the face region slightly (Haar is often tight)
            padding = 10
            x, y = max(0, x - padding), max(0, y - padding)
            w, h = w + 2*padding, h + 2*padding
            
            face = img[y:y+h, x:x+w]
            if face.size > 0:
                face = cv2.resize(face, (160, 160)) # FaceNet expects 160x160
                return face
        return None

    def get_embedding(self, face_img):
        """Get FaceNet embedding for a face image."""
        # face_img should be 160x160 RGB
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        face_img = face_img.astype('float32')
        
        # Add batch dimension
        face_img = np.expand_dims(face_img, axis=0)
        
        # Extract embedding
        yhat = self.facenet.embeddings(face_img)
        return yhat[0]

    def draw_face_info(self, img, bbox, name, confidence):
        """Draw bounding box and label for a detected face."""
        x, y, w, h = bbox
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        
        cv2.rectangle(img, (x, y), (x+w, y+h), color, 2)
        label = f"{name} ({confidence:.2f})"
        cv2.putText(img, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
