import json
import uuid
import os
import math


per_dur = 705600000 # 每拍在sv的时长
time_per_frame = 0.02 # 每帧的时间 hop_size / sample_rate


def build_svp(template, midis, arguments, tempo, basename, extract_pitch, extract_tension, extract_breathiness) -> str:
    notes = [] # 用于保存的音符数据
    datas = [] # 用于记录的音符数据
    new_uuid = str(uuid.uuid4()).lower()

    per_time = 60 / tempo # 每拍的时间
    template["time"]["tempo"] = [{"position": 0, "bpm": tempo}]

    index = 0
    for midi in midis:
        offset = int(arguments[index]["offset"] / per_time * per_dur) # 音符的起始时间在sv的时长

        dur = midi["note_dur"] # 音符的时长
        pitch = midi["note_midi"] # 音符的音高
        rest = midi["note_rest"] # 是否为休止符
        midi_duration = 0 # 该段音符的总时长

        for i in range(len(pitch)):
            current_duration = dur[i] / per_time * per_dur # 当前音符在sv的时长
            onset = midi_duration + offset # 音符的起始时间
            midi_duration += int(current_duration)
            if rest[i]: # 休止符
                continue
            current_pitch = round(pitch[i])

            note = {
                "musicalType": "singing",
                "onset": int(onset),
                "duration": int(current_duration),
                "lyrics": "la",
                "phonemes": "",
                "accent": "",
                "pitch": int(current_pitch),
                "detune": 0,
                "instantMode": False,
                "attributes": {"evenSyllableDuration": True},
                "systemAttributes": {"evenSyllableDuration": True},
                "pitchTakes": {"activeTakeId": 0,"takes": [{"id": 0,"expr": 0,"liked": False}]},
                "timbreTakes": {"activeTakeId": 0,"takes": [{"id": 0,"expr": 0,"liked": False}]}
            }
            notes.append(note)

            data = {"start": int(onset),"finish": int(current_duration + onset),"pitch": int(current_pitch)}
            datas.append(data)
        index += 1

    template["tracks"][0]["mainGroup"]["notes"] = notes
    template["tracks"][0]["mainGroup"]["uuid"] = new_uuid
    template["tracks"][0]["mainRef"]["groupID"] = new_uuid

    pitch, tension, breathiness = [], [], []

    if extract_pitch:
        pitch = build_pitch(datas, arguments, tempo)
    template["tracks"][0]["mainGroup"]["parameters"]["pitchDelta"]["points"] = pitch

    if extract_tension:
        tension = build_arguments(arguments, "tension", tempo)
    template["tracks"][0]["mainGroup"]["parameters"]["tension"]["points"] = tension

    if extract_breathiness:
        breathiness = build_arguments(arguments, "breathiness", tempo)
    template["tracks"][0]["mainGroup"]["parameters"]["breathiness"]["points"] = breathiness

    if extract_pitch and not extract_tension and not extract_breathiness:
        template["tracks"][0]["mainRef"]["voice"]["dF0Vbr"] = 0

    file_path = os.path.join("results", f"{basename}.svp")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(template, f)

    return file_path


def build_pitch(datas: list, arguments: list, tempo: int) -> list:
    pitch = [] # 用于保存的音高数据
    per_time = 60 / tempo # 每拍的时间

    for f0 in arguments:
        offset = f0["offset"]
        f0_data = f0["f0"]
        for i in range(len(f0_data)):
            pitch_onset, pitch_cents = None, None
            f0_value = f0_data[i] # 当前帧的f0值
            if f0_value == 0.0:
                continue
            onset_time = offset + i * time_per_frame # 当前帧的起始时间
            onset = (onset_time / per_time) * per_dur # 当前帧的起始时间在sv的时长
            pitch_onset = int(onset)
            for data in datas:
                if data["start"] <= onset and onset < data["finish"]: # 当前帧在音符的时间范围内
                    pitch_cents = calculate_cents_difference(data["pitch"], f0_value)
                    break
            if pitch_onset is None or pitch_cents is None:
                continue
            pitch.append(pitch_onset)
            pitch.append(pitch_cents)
    return pitch


def build_arguments(arguments: dict, data: str, tempo: int, rate=1.0, argu_offset=0.0):
    args = []
    per_time = 60 / tempo # 每拍的时间

    for arg in arguments:
        offset = arg["offset"]
        datas = arg[data]
        for i in range(len(datas)):
            onset_time = offset + i * time_per_frame # 当前帧的起始时间
            onset = (onset_time / per_time) * per_dur # 当前帧的起始时间在sv的时长
            args.append(onset)
            args.append(datas[i] * rate + argu_offset)
    return args


def calculate_cents_difference(midi_note, f0):
    def midi_to_freq(midi_note):
        A4 = 440.0
        return A4 * (2 ** ((midi_note - 69) / 12))

    def cents_difference(f0, midi_note):
        midi_freq = midi_to_freq(midi_note)
        return 1200 * math.log2(f0 / midi_freq)

    cents_diff = cents_difference(f0, midi_note)
    return round(cents_diff, 5)
