# app.py

import os
import uuid
from flask import (Flask, request, jsonify, render_template, redirect, url_for,
                   flash)
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import google.generativeai as genai
import librosa
import numpy as np
from dotenv import load_dotenv

from flask_sqlalchemy import SQLAlchemy
from flask_login import (LoginManager, UserMixin, login_user, logout_user,
                         login_required, current_user)

# --- Configuration ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Gemini API key not found. Please set it as an environment variable.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Flask App Initialization & Configuration ---
app = Flask(__name__)

# MODIFIED: Load SECRET_KEY from environment variables for security.
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
if not app.config['SECRET_KEY']:
    raise ValueError("SECRET_KEY not found. Please set it as an environment variable.")

# MODIFIED: Configure database path for Render's persistent disk.
# The disk will be mounted at '/var/data', and we'll store the db inside.
# We also ensure the 'instance' directory exists.
PERSISTENT_DISK_DIR = '/var/data'
INSTANCE_PATH = os.path.join(PERSISTENT_DISK_DIR, 'instance')
os.makedirs(INSTANCE_PATH, exist_ok=True)
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{os.path.join(INSTANCE_PATH, "users.db")}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# MODIFIED: UPLOAD_FOLDER should be a temporary directory.
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# --- Database and Login Manager Setup ---
db = SQLAlchemy(app)
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'


# --- User Database Model ---
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(150), unique=True, nullable=False)
    password_hash = db.Column(db.String(256))

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# --- Authentication Routes (Unchanged) ---
@app.route('/login', methods=['GET', 'POST'])
def login():
    if current_user.is_authenticated:
        return redirect(url_for('index'))
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        user = User.query.filter_by(username=username).first()
        if user and user.check_password(password):
            login_user(user)
            return redirect(url_for('index'))
        else:
            flash('Invalid username or password.')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('login'))


# --- Helper Function (Unchanged) ---
def build_gemini_prompt(data):
    base_prompt = (
        "You are an expert music critic. Analyze the following musical submission. "
        "Provide clear, constructive feedback. Use markdown for emphasis (e.g., **key point** or *word*). "
        "Structure your response EXACTLY as follows, using these specific markdown headings:\n"
        "## Overall Score\n"
        "(Your score out of 100 here)\n"
        "(A one-sentence summary explaining the score)\n\n"
        "## Detailed Analysis\n"
        "(Your detailed critique here)\n\n"
        "## Actionable Suggestions\n"
        "(Your specific tips for improvement here)"
    )
    vocal_instruction = ""
    if data.get("vocals_present"):
        vocal_instruction = (
            "The user has indicated this track includes vocals. "
            "In your 'Detailed Analysis' section, please dedicate a paragraph to the vocal performance, commenting on pitch, timing, and emotional delivery."
        )
    top_notes_instruction = ""
    if data.get("top_notes"):
        top_notes_str = ", ".join(data["top_notes"])
        top_notes_instruction = (
            f"The top 3 most prominent pitch classes in this performance were **{top_notes_str}**. "
            "In your 'Detailed Analysis', briefly comment on why these notes might be significant to the piece's key or harmony."
        )
    if data["type"] == "Cover Song":
        return (
            f"{base_prompt}\n\n"
            f"**Submission Type:** Cover Song\n"
            f"**Instrument(s):** {data['instrument']}\n"
            f"**Song:** '{data['song']}' by {data['artist_or_genre']}\n"
            f"**Extracted Data:** The estimated average tempo is {data['tempo']}.\n"
            f"{top_notes_instruction}\n"
            f"{vocal_instruction}"
        )
    elif data["type"] == "Original Composition":
        return (
            f"{base_prompt}\n\n"
            f"**Submission Type:** Original Composition\n"
            f"**Instrument(s):** {data['instrument']}\n"
            f"**Title of Piece:** '{data['song']}'\n"
            f"**Genre/Mood:** {data['artist_or_genre']}\n"
            f"**Extracted Data:** The estimated average tempo is {data['tempo']}.\n"
            f"{top_notes_instruction}\n"
            f"{vocal_instruction}"
        )
    else:  # Remix
        return (
            f"{base_prompt}\n\n"
            f"**Submission Type:** Remix\n"
            f"**Original Song:** '{data['song']}'\n"
            f"**Remix Genre:** {data['artist_or_genre']}\n"
            f"**Extracted Data:** The new average tempo is {data['tempo']}.\n"
            f"{top_notes_instruction}\n"
            f"{vocal_instruction}"
        )


# --- Protected API Endpoint (Unchanged) ---
@app.route('/analyze', methods=['POST'])
@login_required
def analyze_music():
    if 'audio_file' not in request.files: return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files['audio_file']
    if audio_file.filename == '': return jsonify({"error": "No selected file"}), 400
    required_fields = ['submission_type', 'instruments', 'song', 'artist_or_genre']
    if not all(field in request.form for field in required_fields): return jsonify(
        {"error": "Missing one or more required fields"}), 400
    unique_id = str(uuid.uuid4())
    filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
    audio_file.save(audio_path)
    try:
        y, sr = librosa.load(audio_path, sr=None)
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        radar_data = np.mean(chroma, axis=1)
        if radar_data.max() > 0:
            radar_data = (radar_data / radar_data.max()) * 100
        radar_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        top_notes_indices = np.argsort(radar_data)[-3:].tolist()
        top_notes_names = [radar_labels[i] for i in top_notes_indices]
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units='time')
        tempo_variation_data = []
        if len(onsets) > 2:
            iois = np.diff(onsets)
            iois = iois[iois > 0.01]
            tempi = 60.0 / iois
            tempi = np.clip(tempi, 40, 240)
            time_stamps = onsets[:len(tempi)]
            tempo_variation_data = np.vstack((time_stamps, tempi)).T.tolist()
        submission_data = {
            "type": request.form['submission_type'],
            "instrument": request.form['instruments'],
            "song": request.form['song'],
            "artist_or_genre": request.form['artist_or_genre'],
            "tempo": f"{tempo.item():.2f} BPM",
            "vocals_present": request.form.get('vocals_present') == 'on',
            "top_notes": top_notes_names
        }
        prompt = build_gemini_prompt(submission_data)
        response = model.generate_content(prompt)
        feedback_text = response.text
        return jsonify({
            "feedback_text": feedback_text,
            "radar_data": radar_data.tolist(),
            "radar_labels": radar_labels,
            "top_notes_indices": top_notes_indices,
            "estimated_tempo": f"{tempo.item():.2f} BPM",
            "tempo_variation_data": tempo_variation_data
        })
    except Exception as e:
        return jsonify({"error": "An error occurred during analysis.", "details": str(e)}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


# --- Protected Index Route (Unchanged) ---
@app.route('/')
@login_required
def index():
    return render_template('index.html')


# REMOVED: The __main__ block has been removed.
# Database creation and user seeding will be handled by a build script.