import librosa
import os
import yaml
import json
import inference
import importlib
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


def get_midi(chunks: list, model_path: str) -> list:
    """
    Args:
        chunks (list): results from audio_slicer

    Returns:
        list of dict: [{
            "note_midi": array of float, dtype=float32,
            "note_dur": array of float,
            "note_rest": array of bool,
        }, ...]
    """
    infer_cls = inference.task_inference_mapping[config['task_cls']]
    pkg = ".".join(infer_cls.split(".")[:-1])
    cls_name = infer_cls.split(".")[-1]
    infer_cls = getattr(importlib.import_module(pkg), cls_name)
    assert issubclass(infer_cls, inference.BaseInference), \
        f'Inference class {infer_cls} is not a subclass of {inference.BaseInference}.'
    infer_ins = infer_cls(config=config, model_path=model_path)
    midis = infer_ins.infer([c['waveform'] for c in chunks])
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
    midis = get_midi(chunks, model_path)
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
    parser.add_argument('--model_path', type=str, default="weights/model_steps_64000_simplified.ckpt", help='Path to the model file, default: weights/model_steps_64000_simplified.ckpt')
    parser.add_argument('--tempo', type=float, default=120.0, help='Tempo value for the midi file, default: 120')
    parser.add_argument('--extract_pitch', action='store_true', help='Whether to extract pitch from the audio file, default: False')
    parser.add_argument('--extract_tension', action='store_true', help='Whether to extract tension from the audio file, default: False')
    parser.add_argument('--extract_breathiness', action='store_true', help='Whether to extract breathiness from the audio file, default: False')
    args = parser.parse_args()

    assert os.path.isfile("weights/rmvpe.pt"), "RMVPE model not found"
    assert os.path.isfile(args.model_path), "SOME Model not found"

    wav2svp(args.audio_path, args.model_path, args.tempo, args.extract_pitch, args.extract_tension, args.extract_breathiness)