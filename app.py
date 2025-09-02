# app.py

import os
import uuid
import hashlib
import threading
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
# MODIFICATION: Changed model to gemini-2.5-flash-lite as requested
model = genai.GenerativeModel('gemini-2.5-flash-lite')

# --- Flask App Initialization & Configuration ---
app = Flask(__name__)

# In-memory stores for caching and tracking background jobs
analysis_cache = {}
background_jobs = {}

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY')
if not app.config['SECRET_KEY']:
    raise ValueError("SECRET_KEY not found. Please set it as an environment variable.")

# Configure database path for the persistent volume on Fly.io
DATA_DIR = '/data'
DB_PATH = os.path.join(DATA_DIR, 'users.db')
app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{DB_PATH}'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Configure a temporary folder for file uploads
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


# --- Helper Function to Build Prompt for Gemini ---
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


# --- Background Task for Heavy Processing ---
def run_analysis_in_background(job_id, audio_path, submission_data_in, file_hash):
    """This function contains the heavy work and is run in a separate thread."""
    try:
        # 1. Perform audio analysis (CPU intensive)
        # MODIFICATION: Re-enabled downsampling to prevent timeouts and 500 errors.
        sr_target = 44100
        y, sr = librosa.load(audio_path, sr=sr_target)

        # Overall Tempo
        tempo, _ = librosa.beat.beat_track(y=y, sr=sr)

        # Harmony/Pitch Class Prominence (Radar Chart)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        radar_data = np.mean(chroma, axis=1)
        if radar_data.max() > 0:
            radar_data = (radar_data / radar_data.max()) * 100
        radar_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
        top_notes_indices = np.argsort(radar_data)[-3:].tolist()
        top_notes_names = [radar_labels[i] for i in top_notes_indices]

        # Tempo Variation Over Time
        onset_env = librosa.onset.onset_detect(y=y, sr=sr)
        if len(y) > sr * 120:
            y_tempo = y[:sr*120]
        else:
            y_tempo = y
        c_tempo = librosa.beat.tempo(y=y_tempo, sr=sr, aggregate=None)
        times = librosa.times_like(c_tempo, sr=sr)
        tempo_variation_data = [{"x": t, "y": b} for t, b in zip(times, c_tempo)]

        # Pitch Class Prominence Over Time
        frame_length = 2048
        hop_length = 512
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=hop_length, n_fft=frame_length)
        dominant_pitch_indices = np.argmax(chromagram, axis=0)
        dominant_pitch_data = []
        for i, idx in enumerate(dominant_pitch_indices):
            time_point = librosa.frames_to_time(i, sr=sr, hop_length=hop_length)
            dominant_pitch_data.append({"x": time_point, "y": radar_labels[idx]})

        submission_data = submission_data_in.copy()
        submission_data["tempo"] = f"{tempo.item():.2f} BPM"
        submission_data["top_notes"] = top_notes_names

        # 2. Call the external API (Network intensive)
        prompt = build_gemini_prompt(submission_data)
        response = model.generate_content(prompt)
        feedback_text = response.text

        # 3. Store the final result
        result = {
            "feedback_text": feedback_text,
            "radar_data": radar_data.tolist(),
            "radar_labels": radar_labels,
            "top_notes_indices": top_notes_indices,
            "estimated_tempo": f"{tempo.item():.2f} BPM",
            "tempo_variation_data": tempo_variation_data,
            "dominant_pitch_data": dominant_pitch_data,
            "pitch_labels": radar_labels
        }

        analysis_cache[file_hash] = result
        background_jobs[job_id] = {"status": "complete", "result": result}

    except Exception as e:
        print(f"Error in background task for job {job_id}: {e}")
        background_jobs[job_id] = {"status": "error", "message": str(e)}
    finally:
        # 4. Clean up the temporary audio file
        if os.path.exists(audio_path):
            os.remove(audio_path)

# --- ROUTES START HERE ---
@app.route('/analyze', methods=['POST'])
@login_required
def analyze_music():
    if 'audio_file' not in request.files: return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files['audio_file']
    if audio_file.filename == '': return jsonify({"error": "No selected file"}), 400

    audio_bytes = audio_file.read()
    file_hash = hashlib.md5(audio_bytes).hexdigest()
    if file_hash in analysis_cache:
        return jsonify({"status": "cached", "result": analysis_cache[file_hash]})

    audio_file.seek(0)
    unique_id = str(uuid.uuid4())
    filename = secure_filename(audio_file.filename)
    audio_path = os.path.join(app.config['UPLOAD_FOLDER'], f"{unique_id}_{filename}")
    audio_file.save(audio_path)

    submission_data = {
        "type": request.form['submission_type'],
        "instrument": request.form['instruments'],
        "song": request.form['song'],
        "artist_or_genre": request.form['artist_or_genre'],
        "vocals_present": request.form.get('vocals_present') == 'on',
    }
    job_id = unique_id
    background_jobs[job_id] = {"status": "processing"}
    thread = threading.Thread(target=run_analysis_in_background, args=(job_id, audio_path, submission_data, file_hash))
    thread.start()
    return jsonify({"status": "processing", "job_id": job_id})


@app.route('/status/<job_id>')
@login_required
def get_status(job_id):
    job = background_jobs.get(job_id, {"status": "not_found"})
    return jsonify(job)


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


@app.route('/')
@login_required
def index():
    return render_template('index.html')

