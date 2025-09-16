# predict_autism.py — LOAD & PREDICT ONLY (NO TIPS)
import os
import cv2
import numpy as np
import joblib
import mediapipe as mp

# Load the trained model
clf = joblib.load("autism_model.pkl")
print("✅ Loaded trained model.")

# Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Prediction function
def predict_image(image_path):
    print(f"\n🔍 Predicting: {image_path}")
    if not os.path.exists(image_path):
        print("❌ File not found.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print("❌ Cannot read image.")
        return

    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb)

    if not result.multi_face_landmarks:
        print("❌ No face detected.")
        return

    face_landmarks = result.multi_face_landmarks[0]
    landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
    flat = np.array(landmarks).flatten().reshape(1, -1)

    if flat.shape[1] != clf.n_features_in_:
        print(f"❌ Feature mismatch! Model expects {clf.n_features_in_}, but got {flat.shape[1]}")
        return

    prediction = clf.predict(flat)[0]
    confidence = clf.predict_proba(flat)[0][prediction]
    label = "Autistic" if prediction == 1 else "Non-Autistic"

    print(f"✅ Prediction: {label} (Confidence: {confidence:.2%})")

# Predict
predict_image("21.jpg")  # Change to your image path
