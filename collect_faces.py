import cv2
import os
import time

def collect_faces():
    # Initialize OpenCV Haar Cascade for face detection
    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    face_cascade = cv2.CascadeClassifier(cascade_path)

    # Input name of the person
    name = input("Enter the name of the person: ").strip()
    if not name:
        print("Name cannot be empty.")
        return

    # Create directory for the person
    save_path = os.path.join('data', name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    else:
        print(f"Directory for {name} already exists. Appending to it.")

    cap = cv2.VideoCapture(0)
    count = 0
    max_samples = 50

    print(f"Starting collection for {name}. Please look at the camera.")
    print("Press 'q' to quit early.")

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        # Draw face detections
        for (x, y, w, h) in faces:
            # Pad the face region slightly
            padding = 10
            xp, yp = max(0, x - padding), max(0, y - padding)
            wp, hp = w + 2*padding, h + 2*padding
            
            # Crop and save face
            face_img = image[yp:yp+hp, xp:xp+wp]
            if face_img.size > 0:
                count += 1
                file_name = os.path.join(save_path, f"{name}_{count}_{int(time.time())}.jpg")
                # Save as 160x160 to match FaceNet expectation directly if needed
                face_img_resized = cv2.resize(face_img, (160, 160))
                cv2.imwrite(file_name, face_img_resized)
                
                # Draw on display
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, f"Samples: {count}/{max_samples}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # We only collect one face per frame for simplicity
            break

        cv2.imshow('Face Collection', image)

        if count >= max_samples or cv2.waitKey(5) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(f"Successfully collected {count} samples for {name}.")

if __name__ == "__main__":
    collect_faces()
