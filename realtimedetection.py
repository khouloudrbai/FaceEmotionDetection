import cv2
import numpy as np
from keras.models import model_from_json
import sys
sys.stdout.reconfigure(encoding='utf-8')
# Load the model
json_file = open("facialemotionmodel.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("facialemotionmodel.h5")

# Load the Haar Cascade
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}

def extract_features(image):
    feature = np.array(image)
    feature = feature.reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_emotion():
    webcam = cv2.VideoCapture(0)
    _, frame = webcam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    emotion_detected = "No face detected"
    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (48, 48))
        img = extract_features(face)
        pred = model.predict(img)
        emotion_detected = labels[pred.argmax()]
        break  # Process only the first face

    # Save a screenshot of the frame
    cv2.imwrite("screenshot.png", frame)
    webcam.release()
    return emotion_detected

if __name__ == "__main__":
    detected_emotion = detect_emotion()
    print(detected_emotion)
