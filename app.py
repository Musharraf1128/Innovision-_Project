from flask import Flask, render_template, request, send_file, flash
import os
from werkzeug.utils import secure_filename
import pretty_midi
import librosa
import numpy as np
from scipy.signal import find_peaks

app = Flask(__name__)
app.secret_key = 'your_secret_key_here'  # Required for flashing messages

# Configure upload folder
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav', 'mp3', 'ogg', 'm4a'}

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def audio_to_midi(audio_path, output_path, min_note_duration=0.1):
    try:
        # Load the audio file
        audio, sr = librosa.load(audio_path)
        
        # Perform pitch detection
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        
        # Create a new MIDI object
        pm = pretty_midi.PrettyMIDI()
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)
        
        # Convert detected pitches to MIDI notes
        onset_frames = librosa.onset.onset_detect(y=audio, sr=sr)
        onset_times = librosa.frames_to_time(onset_frames, sr=sr)
        
        for i in range(len(onset_times)-1):
            start_idx = librosa.time_to_frames(onset_times[i], sr=sr)
            end_idx = librosa.time_to_frames(onset_times[i+1], sr=sr)
            
            pitch_segment = pitches[:, start_idx:end_idx]
            mag_segment = magnitudes[:, start_idx:end_idx]
            
            if mag_segment.size > 0:
                max_mag_idx = np.unravel_index(mag_segment.argmax(), mag_segment.shape)
                pitch = pitch_segment[max_mag_idx[0], max_mag_idx[1]]
                
                midi_note = librosa.hz_to_midi(pitch)
                note_duration = onset_times[i+1] - onset_times[i]
                
                if note_duration >= min_note_duration:
                    note = pretty_midi.Note(
                        velocity=100,
                        pitch=int(midi_note),
                        start=onset_times[i],
                        end=onset_times[i+1]
                    )
                    piano.notes.append(note)
        
        pm.instruments.append(piano)
        pm.write(output_path)
        return True
    
    except Exception as e:
        print(f"Error in conversion: {str(e)}")
        return False

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if file was uploaded
        if 'file' not in request.files:
            flash('No file selected')
            return render_template('index.html')
        
        file = request.files['file']
        
        # Check if file was actually selected
        if file.filename == '':
            flash('No file selected')
            return render_template('index.html')
        
        if file and allowed_file(file.filename):
            # Secure the filename
            filename = secure_filename(file.filename)
            audio_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            midi_filename = os.path.splitext(filename)[0] + '.mid'
            midi_path = os.path.join(app.config['UPLOAD_FOLDER'], midi_filename)
            
            # Save the uploaded file
            file.save(audio_path)
            
            # Convert to MIDI
            if audio_to_midi(audio_path, midi_path):
                # Clean up the audio file
                os.remove(audio_path)
                
                # Return the MIDI file
                return send_file(
                    midi_path,
                    as_attachment=True,
                    download_name=midi_filename,
                    mimetype='audio/midi'
                )
            else:
                flash('Error converting file')
                return render_template('index.html')
        else:
            flash('Invalid file type')
            return render_template('index.html')
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)