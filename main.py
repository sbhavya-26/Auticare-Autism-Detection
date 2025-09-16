
# main.py — TRAIN & SAVE MODEL
import os
import cv2
import io
import numpy as np
import joblib
from urllib.request import urlopen
from zipfile import ZipFile
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import mediapipe as mp

# Step 1: Download and load dataset
dataset_url = 'https://github.com/Abhirambs-08/05_Early-Prediction-of-Autism-Disorder/archive/refs/heads/main.zip'

print("📥 Downloading dataset...")
with urlopen(dataset_url) as zipresp:
    with ZipFile(io.BytesIO(zipresp.read())) as zfile:
        file_list = [f for f in zfile.namelist() if f.startswith('05_Early-Prediction-of-Autism-Disorder-main/AutismDataset/')]
        images = {}
        for file in file_list:
            if file.endswith('.jpg') or file.endswith('.png'):
                with zfile.open(file) as imagefile:
                    images[file] = imagefile.read()
print(f"✅ Loaded {len(images)} images into memory.")

# Step 2: Initialize MediaPipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)

# Step 3: Extract facial landmarks
landmarks_data = []
for path, bytes_data in images.items():
    np_arr = np.frombuffer(bytes_data, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img is None:
        continue
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(rgb_img)
    if not result.multi_face_landmarks:
        continue
    face_landmarks = result.multi_face_landmarks[0]
    landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
    landmarks_data.append({'file_path': path, 'landmarks': landmarks})
print(f"📊 Total usable images: {len(landmarks_data)}")

# Step 4: Get labels
def get_label(path):
    if '/Autistic/' in path:
        return 1
    elif '/NonAutistic/' in path or '/Non_Autistic/' in path:
        return 0
    fname = path.split('/')[-1].lower()
    if fname.startswith('autistic'):
        return 1
    elif fname.startswith('nonautistic') or fname.startswith('non_autistic'):
        return 0
    return None

features, labels = [], []
for data in landmarks_data:
    label = get_label(data['file_path'])
    if label is not None:
        features.append(np.array(data['landmarks']).flatten())
        labels.append(label)

features = np.array(features)
labels = np.array(labels)

print("✅ Features shape:", features.shape)
print("✅ Labels shape:", labels.shape)

# Step 5: Train & evaluate model
X_train, X_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print("\n📈 Classification Report:")
print(classification_report(y_test, y_pred))
print(f"✅ Accuracy: {accuracy_score(y_test, y_pred):.4f}")

# Step 6: Save model
joblib.dump(clf, "autism_model.pkl")
print("💾 Model saved as autism_model.pkl")