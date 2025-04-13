
from flask import Flask, request, render_template, send_from_directory, redirect, url_for
import os
import pickle
import face_recognition
from werkzeug.utils import secure_filename
from PIL import Image
import shutil

app = Flask(__name__)

# === CONFIGURATION ===
UPLOAD_FOLDER = 'event_photos'
GUEST_FOLDER = 'guest_photos'
MATCHED_FOLDER = 'matched_photos'
GUEST_DATA = 'guest_data.pkl'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

for folder in [UPLOAD_FOLDER, GUEST_FOLDER, MATCHED_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# === UTILS ===
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

# === ROUTES ===
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

        image = face_recognition.load_image_file(filepath)
        encodings = face_recognition.face_encodings(image)
        if not encodings:
            return "No face detected in selfie.", 400

        guest_data = load_guest_data()
        guest_data[name] = encodings[0]
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
    shutil.rmtree(MATCHED_FOLDER)
    os.makedirs(MATCHED_FOLDER, exist_ok=True)

    for img_file in os.listdir(UPLOAD_FOLDER):
        img_path = os.path.join(UPLOAD_FOLDER, img_file)
        image = face_recognition.load_image_file(img_path)
        locations = face_recognition.face_locations(image)
        encodings = face_recognition.face_encodings(image, locations)

        for name, guest_encoding in guest_data.items():
            matches = face_recognition.compare_faces(encodings, guest_encoding, tolerance=0.5)
            if any(matches):
                guest_folder = os.path.join(MATCHED_FOLDER, name)
                os.makedirs(guest_folder, exist_ok=True)
                shutil.copy(img_path, os.path.join(guest_folder, img_file))
    return "Matching complete."

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
    app.run(debug=True)
