import os
import pickle
import numpy as np
import cv2
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from utils import FaceRecognitionUtils

def train():
    utils = FaceRecognitionUtils()
    data_dir = 'data'
    X = [] # Embeddings
    y = [] # Labels

    print("Extracting embeddings from dataset...")
    for person_name in os.listdir(data_dir):
        person_dir = os.path.join(data_dir, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            # In our collect_faces.py, the images are already cropped faces,
            # but we'll use extract_face again to be sure or just resize.
            # FaceNet expects 160x160.
            face = cv2.resize(img, (160, 160))
            embedding = utils.get_embedding(face)
            
            X.append(embedding)
            y.append(person_name)

    if not X:
        print("No faces found in data directory. Please run collect_faces.py first.")
        return

    X = np.asarray(X)
    y = np.asarray(y)

    print(f"Training SVM on {len(X)} samples for {len(set(y))} people...")
    
    # Label encoding
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Train SVM
    model = SVC(kernel='linear', probability=True)
    model.fit(X, y_encoded)

    # Save model and label encoder
    models_dir = 'models'
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    with open(os.path.join(models_dir, 'face_recognition_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
    
    with open(os.path.join(models_dir, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(le, f)

    print("Training complete! Model saved in models/ directory.")

if __name__ == "__main__":
    train()
