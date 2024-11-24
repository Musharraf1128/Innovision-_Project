import librosa
import pretty_midi
import numpy as np
from scipy.signal import find_peaks
import sys
import os

def audio_to_midi(audio_path, output_path, min_note_duration=0.1):
    """
     I have used ai's help to make this, so, not entirely written by me
     just a note on how to run this 
     use these command on your command line:
     if on linux : python3 audio-to-midi.py yourInputAudioFileName.wav yourOutputFileName.mid
     if on windows : python audio-to-midi.py yourInputAudioFileName.wav yourOutputFileName.mid
    """
    try:
        # Check if input file exists
        if not os.path.exists(audio_path):
            print(f"Error: Input file '{audio_path}' not found.")
            return

        print(f"Loading audio file: {audio_path}")
        audio, sr = librosa.load(audio_path)
        
        print("Detecting pitches...")
        pitches, magnitudes = librosa.piptrack(y=audio, sr=sr)
        
        # Create a new MIDI object
        pm = pretty_midi.PrettyMIDI()
        piano_program = pretty_midi.instrument_name_to_program('Acoustic Grand Piano')
        piano = pretty_midi.Instrument(program=piano_program)
        
        print("Converting to MIDI notes...")
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
        
        print(f"Saving MIDI file to: {output_path}")
        pm.write(output_path)
        print("Conversion completed successfully!")
        
        return pm
    
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return None

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 audio-to-midi.py <input_audio_file> [output_midi_file]")
        sys.exit(1)
    
    input_file = sys.argv[1]
    
    output_file = sys.argv[2] if len(sys.argv) > 2 else "output.mid"
    
    audio_to_midi(input_file, output_file)

if __name__ == "__main__":
    main()
