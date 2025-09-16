from flask import Flask, render_template, request, redirect, url_for
import os
import cv2
import numpy as np
import pandas as pd
import joblib
import mediapipe as mp
from flask_bcrypt import Bcrypt
from flask_login import LoginManager, UserMixin, login_user, logout_user, login_required, current_user

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['UPLOAD_FOLDER'] = 'static/uploads'

bcrypt = Bcrypt(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# Load trained model
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

class User(UserMixin):
    def __init__(self, id_, username):
        self.id = id_
        self.username = username

@login_manager.user_loader
def load_user(user_id):
    users = pd.read_csv('users.csv') if os.path.exists('users.csv') else pd.DataFrame()
    user_row = users[users['id'] == int(user_id)]
    if not user_row.empty:
        return User(user_row.iloc[0]['id'], user_row.iloc[0]['username'])
    return None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users_file = 'users.csv'
        users = pd.read_csv(users_file) if os.path.exists(users_file) else pd.DataFrame(columns=['id', 'username', 'password'])

        if username in users['username'].values:
            return "Username already exists!"

        hashed_pw = bcrypt.generate_password_hash(password).decode('utf-8')
        new_id = users['id'].max() + 1 if not users.empty else 1
        users = pd.concat([users, pd.DataFrame([{'id': new_id, 'username': username, 'password': hashed_pw}])])
        users.to_csv(users_file, index=False)
        return redirect(url_for('login'))
    return render_template('signup.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        users = pd.read_csv('users.csv') if os.path.exists('users.csv') else pd.DataFrame()
        user_row = users[users['username'] == username]
        if not user_row.empty and bcrypt.check_password_hash(user_row.iloc[0]['password'], password):
            user = User(user_row.iloc[0]['id'], username)
            login_user(user)
            return redirect(url_for('predict_page'))
        return "Invalid credentials!"
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('index'))

@app.route('/predict_page')
@login_required
def predict_page():
    return render_template('predict.html')

@app.route('/predict', methods=['POST'])
@login_required
def predict():
    if 'file' not in request.files:
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        return redirect(request.url)

    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        img = cv2.imread(file_path)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            label = "No face detected"
            confidence = "N/A"
        else:
            face_landmarks = result.multi_face_landmarks[0]
            landmarks = [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            flat = np.array(landmarks).flatten().reshape(1, -1)

            expected_length = clf.n_features_in_
            current_length = flat.shape[1]
            if current_length < expected_length:
                flat = np.pad(flat, ((0, 0), (0, expected_length - current_length)), 'constant')
            elif current_length > expected_length:
                flat = flat[:, :expected_length]

            prediction = clf.predict(flat)[0]
            confidence_score = clf.predict_proba(flat)[0][prediction]
            label = "Autistic" if prediction == 1 else "Non-Autistic"
            confidence = f"{confidence_score:.2%}"

            history_file = "prediction_history.csv"
            history = pd.DataFrame([{
                'Filename': filename,
                'Prediction': label,
                'Confidence': confidence
            }])
            if os.path.exists(history_file):
                history.to_csv(history_file, mode='a', header=False, index=False)
            else:
                history.to_csv(history_file, index=False)

        return render_template(
            'result.html',
            label=label,
            confidence=confidence,
            image_url=url_for('static', filename='uploads/' + filename)
        )

@app.route('/history')
@login_required
def history():
    history_file = "prediction_history.csv"
    if os.path.exists(history_file):
        history = pd.read_csv(history_file)
        records = history.to_dict(orient='records')
    else:
        records = []
    return render_template('history.html', history=records)

@app.route('/delete_record/<int:row_id>', methods=['POST'])
@login_required
def delete_record(row_id):
    history_file = "prediction_history.csv"
    if os.path.exists(history_file):
        history = pd.read_csv(history_file)
        if 0 <= row_id < len(history):
            history = history.drop(index=row_id).reset_index(drop=True)
            history.to_csv(history_file, index=False)
    return redirect(url_for('history'))

if __name__ == '__main__':
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    app.run(debug=True)
