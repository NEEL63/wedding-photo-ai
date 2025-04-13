
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
import shutil
import pickle
from werkzeug.utils import secure_filename
from deepface import DeepFace
from PIL import Image
import numpy as np

app = Flask(__name__)

UPLOAD_FOLDER = 'event_photos'
GUEST_FOLDER = 'guest_photos'
MATCHED_FOLDER = 'matched_photos'
GUEST_DATA = 'guest_data.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

for folder in [UPLOAD_FOLDER, GUEST_FOLDER, MATCHED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_guest_data():
    if os.path.exists(GUEST_DATA):
        with open(GUEST_DATA, 'rb') as f:
            return pickle.load(f)
    return {}

def save_guest_data(data):
    with open(GUEST_DATA, 'wb') as f:
        pickle.dump(data, f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload_selfie', methods=['POST'])
def upload_selfie():
    name = request.form['name']
    file = request.files['selfie']
    if file and allowed_file(file.filename):
        filename = secure_filename(f"{name}.jpg")
        filepath = os.path.join(GUEST_FOLDER, filename)
        file.save(filepath)

        guest_data = load_guest_data()
        guest_data[name] = filepath
        save_guest_data(guest_data)

        return redirect(url_for('home'))
    return "Invalid file format.", 400

@app.route('/upload_event', methods=['POST'])
def upload_event():
    files = request.files.getlist('eventphotos')
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(UPLOAD_FOLDER, filename))
    return redirect(url_for('home'))

@app.route('/match_faces')
def match_faces():
    guest_data = load_guest_data()
    shutil.rmtree(MATCHED_FOLDER, ignore_errors=True)
    os.makedirs(MATCHED_FOLDER, exist_ok=True)

    logs = []

    for photo_name in os.listdir(UPLOAD_FOLDER):
        photo_path = os.path.join(UPLOAD_FOLDER, photo_name)

        for guest_name, guest_photo_path in guest_data.items():
            try:
                result = DeepFace.verify(
                    img1_path=guest_photo_path,
                    img2_path=photo_path,
                    enforce_detection=False,
                    detector_backend='opencv',  # Use lightweight detector
                    model_name='VGG-Face',
                    prog_bar=False,
                    distance_metric='cosine'
                )

                if result['verified']:
                    guest_folder = os.path.join(MATCHED_FOLDER, guest_name)
                    os.makedirs(guest_folder, exist_ok=True)
                    shutil.copy(photo_path, os.path.join(guest_folder, photo_name))
                    logs.append(f"Matched: {guest_name} <-> {photo_name}")
            except Exception as e:
                logs.append(f"Error matching {guest_name} and {photo_name}: {str(e)}")

    print(\"MATCHING LOG:\")
    for log in logs:
        print(log)

    return \"Face matching complete.\"


@app.route('/view_album/<guest_name>')
def view_album(guest_name):
    folder = os.path.join(MATCHED_FOLDER, guest_name)
    if not os.path.exists(folder):
        return "No matched photos found."
    files = os.listdir(folder)
    return render_template('album.html', files=files, guest=guest_name)

@app.route('/matched_photos/<guest>/<filename>')
def matched_photos(guest, filename):
    return send_from_directory(os.path.join(MATCHED_FOLDER, guest), filename)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=10000)

