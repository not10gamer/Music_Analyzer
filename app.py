import os
import uuid
from flask import Flask, request, jsonify, url_for, render_template
from werkzeug.utils import secure_filename
import google.generativeai as genai
import librosa
import numpy as np
from dotenv import load_dotenv

# --- Configuration (Unchanged) ---
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("Gemini API key not found. Please set it in a .env file.")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')

# --- Flask App Initialization (Unchanged) ---
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# --- Helper Function (Unchanged) ---
def build_gemini_prompt(data):
    """Constructs a tailored prompt for the Gemini API."""
    base_prompt = (
        "You are an expert music critic. Analyze the following musical submission. "
        "Provide clear, constructive feedback. Use markdown for emphasis (e.g., **key point**). "
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
            f"**Extracted Data:** The estimated tempo is {data['tempo']}.\n"
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
            f"**Extracted Data:** The estimated tempo is {data['tempo']}.\n"
            f"{top_notes_instruction}\n"
            f"{vocal_instruction}"
        )
    else:  # Remix
        return (
            f"{base_prompt}\n\n"
            f"**Submission Type:** Remix\n"
            f"**Original Song:** '{data['song']}'\n"
            f"**Remix Genre:** {data['artist_or_genre']}\n"
            f"**Extracted Data:** The new tempo is {data['tempo']}.\n"
            f"{top_notes_instruction}\n"
            f"{vocal_instruction}"
        )


# --- API Endpoint (UPDATED) ---
@app.route('/analyze', methods=['POST'])
def analyze_music():
    if 'audio_file' not in request.files: return jsonify({"error": "No audio file provided"}), 400
    audio_file = request.files['audio_file']
    if audio_file.filename == '': return jsonify({"error": "No selected file"}), 400

    # UPDATED: 'instrument' is now 'instruments'
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
        tempo = tempo.item()

        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        radar_data = np.mean(chroma, axis=1)
        if radar_data.max() > 0:
            radar_data = (radar_data / radar_data.max()) * 100

        radar_labels = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']

        top_notes_indices = np.argsort(radar_data)[-3:].tolist()
        top_notes_names = [radar_labels[i] for i in top_notes_indices]

        submission_data = {
            "type": request.form['submission_type'],
            "instrument": request.form['instruments'],  # Changed from 'instrument'
            "song": request.form['song'],
            "artist_or_genre": request.form['artist_or_genre'],
            "tempo": f"{tempo:.2f} BPM",
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
            "estimated_tempo": f"{tempo:.2f} BPM"
        })

    except Exception as e:
        return jsonify({"error": "An error occurred during analysis.", "details": str(e)}), 500
    finally:
        if os.path.exists(audio_path):
            os.remove(audio_path)


# --- Index Route and Main Execution (Unchanged) ---
@app.route('/')
def index():
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0')