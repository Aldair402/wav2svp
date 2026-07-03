import librosa
import os
import yaml
import json
import numpy as np
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

# ========== GAME ==========
# So this is kinda important so im usin == to not forget
import game
from game.infer import extract as game_extract
# ===============================================

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


def get_midi(chunks: list, model_path: str) -> list:
    """
    EXTRAE MIDI USANDO GAME (en lugar de SOME).
    La firma y el formato de salida son IDÉNTICOS al original.

    Args:
        chunks (list): results from audio_slicer

    Returns:
        list of dict: [{
            "note_midi": array of float, dtype=float32,
            "note_dur": array of float,
            "note_rest": array of bool,
        }, ...]
    """
    midis = []
    # This is probably fine? idk i didnt look to much into the code yet
    language = 'zh'  

    for chunk in tqdm(chunks, desc='Extracting MIDI with GAME'):
        waveform_16k = chunk['waveform_16k']  # array float32, 16kHz

        # ========== Call GAME ==========
        # BE SURE TO NOT FUCK THIS UP
        result = game_extract(
            audio=waveform_16k,
            model_path=model_path,
            language=language,
            sr=16000
        )
        notes = result.notes  # lista de tuplas (start, end, pitch), time in sec
        # ====================================

        # Turn GAME output to whatever build_svp expects / build_midi_file ---
        note_midi = []
        note_dur = []
        note_rest = []

        last_end = 0.0
        chunk_dur = len(waveform_16k) / 16000.0

        for start, end, pitch in notes:
            # Silencio antes de esta nota (si hay gap)
            if start > last_end:
                silence_dur = start - last_end
                note_midi.append(0.0)          # valor dummy para silencio
                note_dur.append(silence_dur)
                note_rest.append(True)

            # Notes 
            dur = end - start
            note_midi.append(float(pitch))
            note_dur.append(dur)
            note_rest.append(False)

            last_end = end

        # Silence
        if last_end < chunk_dur:
            silence_dur = chunk_dur - last_end
            note_midi.append(0.0)
            note_dur.append(silence_dur)
            note_rest.append(True)

        # Savee
        midis.append({
            'note_midi': np.array(note_midi, dtype=np.float32),
            'note_dur': np.array(note_dur, dtype=np.float32),
            'note_rest': np.array(note_rest, dtype=bool),
        })

    return midis


def save_midi(midis: list, tempo: int, chunks: list, midi_path: str) -> None:
    midi_file = build_midi_file([c['offset'] for c in chunks], midis, tempo=tempo)
    midi_file.save(midi_path)


def get_f0(chunks: list):
    rmvpe = RMVPE(model_path='weights/rmvpe.pt') # hop_size=160
    for chunk in tqdm(chunks, desc='Extracting F0'):
        chunk['f0'] = rmvpe.infer_from_audio(chunk['waveform_16k'], sample_rate=16000)[::2].astype(float)
    return chunks


def get_energy_librosa(waveform, hop_size, win_size):
    energy = librosa.feature.rms(y=waveform, frame_length=win_size, hop_length=hop_size)[0]
    return energy


def get_breathiness(chunks, hop_size, win_size, sigma=1.0):
    for chunk in tqdm(chunks, desc='Extracting Breathiness'):
        waveform = chunk['waveform_16k']
        waveform_ap = librosa.effects.percussive(waveform)
        breathiness = get_energy_librosa(waveform_ap, hop_size, win_size)
        breathiness = (2 / max(abs(np.max(breathiness)), abs(np.min(breathiness)))) * breathiness
        breathiness = np.tanh(breathiness - np.mean(breathiness))
        breathiness_smoothed = gaussian_filter(breathiness, sigma=sigma)
        chunk['breathiness'] = breathiness_smoothed[::2].astype(float)
    return chunks


def get_tension(chunks, hop_size, win_size, sigma=1.0):
    for chunk in tqdm(chunks, desc='Extracting Tension'):
        waveform = chunk['waveform_16k']
        waveform_h = librosa.effects.harmonic(waveform)
        waveform_base_h = librosa.effects.harmonic(waveform, power=0.5)
        energy_base_h = get_energy_librosa(waveform_base_h, hop_size, win_size)
        energy_h = get_energy_librosa(waveform_h, hop_size, win_size)
        tension = np.sqrt(np.clip(energy_h ** 2 - energy_base_h ** 2, 0, None)) / (energy_h + 1e-5)
        tension = (2 / max(abs(np.max(tension)), abs(np.min(tension)))) * tension
        tension = np.tanh(tension - np.mean(tension))
        tension_smoothed = gaussian_filter(tension, sigma=sigma)
        chunk['tension'] = tension_smoothed[::2].astype(float)
    return chunks


def get_arguments(chunks, hop_size, win_size, extract_pitch=False, extract_tension=False, extract_breathiness=False):
    if extract_pitch:
        chunks = get_f0(chunks)
    if extract_tension:
        chunks = get_tension(chunks, hop_size, win_size)
    if extract_breathiness:
        chunks = get_breathiness(chunks, hop_size, win_size)
    return chunks


def wav2svp(audio_path, model_path, tempo=120, extract_pitch=False, extract_tension=False, extract_breathiness=False):
    os.makedirs('results', exist_ok=True)
    basename = os.path.basename(audio_path).split('.')[0]

    chunks = audio_slicer(audio_path)
    midis = get_midi(chunks, model_path)   # <--- LA FIRMA ES EXACTAMENTE LA MISMA
    arguments = get_arguments(
        chunks, hop_size=160, win_size=1024, 
        extract_pitch=extract_pitch, extract_tension=extract_tension, extract_breathiness=extract_breathiness
    )

    template = load_config('template.json')

    print("building svp file")
    svp_path = build_svp(template, midis, arguments, tempo, basename, extract_pitch, extract_tension, extract_breathiness)

    print("building midi file")
    midi_path = os.path.join('results', f'{basename}.mid')
    save_midi(midis, tempo, chunks, midi_path)

    print("Success")
    return svp_path, midi_path


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Inference for wav2svp')
    parser.add_argument('audio_path', type=str, help='Path to the input audio file')
    # Keep original name!
    parser.add_argument('--model_path', type=str, default="weights/model_steps_64000_simplified.ckpt", help='Path to the model file, default: weights/model_steps_64000_simplified.ckpt')
    parser.add_argument('--tempo', type=float, default=120.0, help='Tempo value for the midi file, default: 120')
    parser.add_argument('--extract_pitch', action='store_true', help='Whether to extract pitch from the audio file, default: False')
    parser.add_argument('--extract_tension', action='store_true', help='Whether to extract tension from the audio file, default: False')
    parser.add_argument('--extract_breathiness', action='store_true', help='Whether to extract breathiness from the audio file, default: False')
    args = parser.parse_args()

    assert os.path.isfile("weights/rmvpe.pt"), "RMVPE model not found"
    assert os.path.isfile(args.model_path), "GAME Model not found"

    wav2svp(args.audio_path, args.model_path, args.tempo, args.extract_pitch, args.extract_tension, args.extract_breathiness)
