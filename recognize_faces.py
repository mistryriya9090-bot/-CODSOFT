import cv2
import pickle
import numpy as np
import os
from utils import FaceRecognitionUtils

def recognize():
    utils = FaceRecognitionUtils()
    models_dir = 'models'
    model_path = os.path.join(models_dir, 'face_recognition_model.pkl')
    le_path = os.path.join(models_dir, 'label_encoder.pkl')

    if not os.path.exists(model_path) or not os.path.exists(le_path):
        print("Training files not found. Please run train_model.py first.")
        return

    # Load model and label encoder
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(le_path, 'rb') as f:
        le = pickle.load(f)

    # Initialize OpenCV Haar Cascade for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    cap = cv2.VideoCapture(0)
    print("Starting face recognition. Press 'q' to quit.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            # Pad the face region slightly (matching training collection)
            padding = 10
            xp, yp = max(0, x - padding), max(0, y - padding)
            wp, hp = w + 2*padding, h + 2*padding
            
            # Crop face
            face_img = image[yp:yp+hp, xp:xp+wp]
            if face_img.size > 0:
                face_img = cv2.resize(face_img, (160, 160))
                embedding = utils.get_embedding(face_img)
                
                # Predict using SVM
                probs = model.predict_proba([embedding])[0]
                best_match_idx = np.argmax(probs)
                confidence = probs[best_match_idx]
                
                # Threshold for "Unknown"
                if confidence > 0.7: # Adjusted threshold for SVM
                    name = le.inverse_transform([best_match_idx])[0]
                else:
                    name = "Unknown"

                # Visualization
                color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
                cv2.rectangle(image, (x, y), (x+w, y+h), color, 2)
                label = f"{name} ({confidence:.2f})"
                cv2.putText(image, label, (x, y-10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        cv2.imshow('Face Recognition', image)

        if cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    recognize()
