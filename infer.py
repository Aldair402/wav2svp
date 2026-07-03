import librosa
import os
import yaml
import json
import subprocess
import tempfile
import shutil
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from modules.rmvpe.inference import RMVPE
from utils.slicer2 import Slicer
from utils.infer_utils import build_midi_file
from build_svp import build_svp

def load_config(config_path: str) -> dict:
    if config_path.endswith('.yaml'):
        with open(config_path, 'r', encoding='utf8') as f:
            config = yaml.safe_load(f)
    elif config_path.endswith('.json'):
        with open(config_path, 'r', encoding='utf8') as f:
            config = json.load(f)
    else:
        raise ValueError(f'Unsupported config file format: {config_path}')
    return config

config = load_config('weights/config.yaml')
sr = config['audio_sample_rate']

def audio_slicer(audio_path: str) -> list:
    """
    Returns:
        list of dict: [{
            "offset": np.float64,
            "waveform": array of float, dtype=float32,
        }, ...]
    """
    waveform, _ = librosa.load(audio_path, sr=sr, mono=True)
    slicer = Slicer(sr=sr, max_sil_kept=1000)
    chunks = slicer.slice(waveform)
    for c in chunks:
        c['waveform_16k'] = librosa.resample(y=c['waveform'], orig_sr=sr, target_sr=16000)
    return chunks

def run_game_extract(audio_path: str, model_path: str) -> dict:
    """
    Ejecuta GAME sobre el archivo de audio y devuelve las rutas a los archivos de salida.
    """
    # Crear un directorio temporal y copiar el audio
    with tempfile.TemporaryDirectory() as tmpdir:
        # Copiar el archivo de audio al temporal
        audio_basename = os.path.basename(audio_path)
        temp_audio = os.path.join(tmpdir, audio_basename)
        shutil.copy2(audio_path, temp_audio)
        
        # Ejecutar game/infer.py sobre el directorio temporal
        cmd = [
            'python', 'game/infer.py', 'extract',
            tmpdir,
            '-m', model_path,
            '--glob', '*.wav',
            '--output-formats', 'mid,txt,csv'
        ]
        # Cambiar al directorio temporal para que los archivos de salida se guarden allí
        original_cwd = os.getcwd()
        os.chdir(tmpdir)
        try:
            result = subprocess.run(
                ['python', os.path.join(original_cwd, 'game/infer.py'), 'extract', '.', '-m', model_path, '--glob', '*.wav', '--output-formats', 'mid,txt,csv'],
                check=True,
                capture_output=True,
                text=True
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Error al ejecutar GAME:\nSTDOUT: {e.stdout}\nSTDERR: {e.stderr}")
        finally:
            os.chdir(original_cwd)
        
        # Buscar los archivos generados (mid, txt, csv) en el directorio temporal
        basename = os.path.splitext(audio_basename)[0]
        mid_file = None
        txt_file = None
        csv_file = None
        for root, dirs, files in os.walk(tmpdir):
            for f in files:
                if f.endswith('.mid'):
                    mid_file = os.path.join(root, f)
                elif f.endswith('.txt'):
                    txt_file = os.path.join(root, f)
                elif f.endswith('.csv'):
                    csv_file = os.path.join(root, f)
        # Si no se encontraron, intentar con el nombre base en el directorio raíz del temporal
        if not mid_file:
            mid_file = os.path.join(tmpdir, f'{basename}.mid')
        if not txt_file:
            txt_file = os.path.join(tmpdir, f'{basename}.txt')
        if not csv_file:
            csv_file = os.path.join(tmpdir, f'{basename}.csv')
        
        # Verificar que existan
        if not os.path.exists(mid_file):
            raise FileNotFoundError(f"No se generó el archivo MIDI en {tmpdir}")
        if not os.path.exists(txt_file) and not os.path.exists(csv_file):
            raise FileNotFoundError(f"No se generó archivo de características (txt/csv) en {tmpdir}")
        
        # Devolver las rutas (los archivos se eliminarán al salir del contexto)
        # Para conservarlos, los copiamos a un directorio persistente (por ejemplo, 'game_outputs')
        # Pero los usaremos inmediatamente, así que devolvemos las rutas temporales.
        return {
            'mid': mid_file,
            'txt': txt_file,
            'csv': csv_file
        }

def parse_game_outputs(output_files: dict, tempo: float):
    """
    Lee los archivos generados por GAME y devuelve:
        notes: lista de tuplas (start, end, pitch, velocity)
        pitch_frames: array de pitch por frame
        tension_frames: array de tensión por frame
        breathiness_frames: array de respiración por frame
    """
    # Parsear el archivo MIDI para obtener las notas
    import mido
    mid = mido.MidiFile(output_files['mid'])
    notes = []
    ticks_per_beat = mid.ticks_per_beat
    tempo_micros = 500000  # valor por defecto (120 BPM)
    for track in mid.tracks:
        for msg in track:
            if msg.type == 'set_tempo':
                tempo_micros = msg.tempo
                break
    sec_per_tick = tempo_micros / 1e6 / ticks_per_beat
    current_time = 0.0
    notes_on = {}
    for track in mid.tracks:
        for msg in track:
            current_time += msg.time * sec_per_tick
            if msg.type == 'note_on' and msg.velocity > 0:
                notes_on[msg.note] = current_time
            elif msg.type == 'note_off' or (msg.type == 'note_on' and msg.velocity == 0):
                if msg.note in notes_on:
                    start = notes_on.pop(msg.note)
                    end = current_time
                    notes.append((start, end, msg.note, 100))
    notes.sort(key=lambda x: x[0])
    
    # Leer archivo CSV o TXT para features
    feature_file = output_files['csv'] if os.path.exists(output_files['csv']) else output_files['txt']
    if not os.path.exists(feature_file):
        raise FileNotFoundError("No se encontró archivo de características de GAME")
    
    import csv
    with open(feature_file, 'r') as f:
        sample = f.read(1024)
        f.seek(0)
        sniffer = csv.Sniffer()
        dialect = sniffer.sniff(sample)
        has_header = sniffer.has_header(sample)
        reader = csv.reader(f, dialect)
        rows = list(reader)
        if has_header:
            header = rows[0]
            rows = rows[1:]
        else:
            header = None
        # Identificar columnas por nombre o posición
        col_time = 0
        col_pitch = 1
        col_tension = 2
        col_breath = 3
        if has_header and header:
            for i, name in enumerate(header):
                if 'time' in name.lower():
                    col_time = i
                elif 'pitch' in name.lower():
                    col_pitch = i
                elif 'tension' in name.lower():
                    col_tension = i
                elif 'breath' in name.lower() or 'breathiness' in name.lower():
                    col_breath = i
        times, pitches, tensions, breaths = [], [], [], []
        for row in rows:
            if len(row) > max(col_time, col_pitch, col_tension, col_breath):
                times.append(float(row[col_time]))
                pitches.append(float(row[col_pitch]))
                tensions.append(float(row[col_tension]))
                breaths.append(float(row[col_breath]))
        times = np.array(times)
        pitches = np.array(pitches)
        tensions = np.array(tensions)
        breaths = np.array(breaths)
    
    return notes, pitches, tensions, breaths

def build_midi_from_notes(notes, tempo):
    from midiutil import MIDIFile
    midi = MIDIFile(1)
    midi.addTempo(0, 0, tempo)
    for start, end, pitch, vel in notes:
        duration = end - start
        midi.addNote(0, 0, pitch, start, duration, vel)
    return midi

def save_midi_from_notes(notes, tempo, midi_path):
    midi = build_midi_from_notes(notes, tempo)
    with open(midi_path, 'wb') as f:
        midi.writeFile(f)

def wav2svp(audio_path, model_path, tempo=120, extract_pitch=False, extract_tension=False, extract_breathiness=False):
    os.makedirs('results', exist_ok=True)
    basename = os.path.basename(audio_path).split('.')[0]
    
    # 1. Ejecutar GAME para extraer MIDI y características
    try:
        output_files = run_game_extract(audio_path, model_path)
        notes, pitch_frames, tension_frames, breath_frames = parse_game_outputs(output_files, tempo)
    except Exception as e:
        raise RuntimeError(f"Error al procesar con GAME: {e}")
    
    # 2. Construir el archivo SVP
    if notes:
        note_starts = np.array([n[0] for n in notes])
        note_ends = np.array([n[1] for n in notes])
        note_pitches = np.array([n[2] for n in notes])
        note_durs = note_ends - note_starts
        note_rest = np.zeros(len(notes), dtype=bool)
    else:
        note_pitches = np.array([])
        note_durs = np.array([])
        note_rest = np.array([])
    
    midis = [{
        'note_midi': note_pitches,
        'note_dur': note_durs,
        'note_rest': note_rest
    }]
    
    # Obtener duración del audio
    duration = librosa.get_duration(filename=audio_path)
    if extract_pitch and len(pitch_frames) > 0:
        pitch_arr = pitch_frames
    else:
        pitch_arr = np.array([])
    if extract_tension and len(tension_frames) > 0:
        tension_arr = tension_frames
    else:
        tension_arr = np.array([])
    if extract_breathiness and len(breath_frames) > 0:
        breath_arr = breath_frames
    else:
        breath_arr = np.array([])
    
    if len(pitch_arr) > 0:
        hop_time = duration / len(pitch_arr)
    else:
        hop_time = 0.01
    
    arguments = [{
        'pitch': pitch_arr,
        'tension': tension_arr,
        'breathiness': breath_arr,
        'hop_time': hop_time
    }]
    
    template = load_config('template.json')
    print("building svp file")
    svp_path = build_svp(template, midis, arguments, tempo, basename, extract_pitch, extract_tension, extract_breathiness)
    
    print("building midi file")
    midi_path = os.path.join('results', f'{basename}.mid')
    save_midi_from_notes(notes, tempo, midi_path)
    
    print("Success")
    return svp_path, midi_path

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Inference for wav2svp using GAME')
    parser.add_argument('audio_path', type=str, help='Path to the input audio file')
    parser.add_argument('--model_path', type=str, default="weights/model_steps_64000_simplified.ckpt", help='Path to the GAME model file')
    parser.add_argument('--tempo', type=float, default=120.0, help='Tempo value for the midi file')
    parser.add_argument('--extract_pitch', action='store_true', help='Whether to extract pitch from the audio file')
    parser.add_argument('--extract_tension', action='store_true', help='Whether to extract tension from the audio file')
    parser.add_argument('--extract_breathiness', action='store_true', help='Whether to extract breathiness from the audio file')
    args = parser.parse_args()
    
    assert os.path.isfile("weights/rmvpe.pt"), "RMVPE model not found"
    assert os.path.isfile(args.model_path), "GAME Model not found"
    assert os.path.isdir("game"), "GAME folder not found in current directory"
    
    wav2svp(args.audio_path, args.model_path, args.tempo, args.extract_pitch, args.extract_tension, args.extract_breathiness)
