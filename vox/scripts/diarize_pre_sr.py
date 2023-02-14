import torch
import matplotlib.pyplot as plt
import json
import os
from pathlib import Path
from statistics import median
from tqdm import tqdm
from pyannote.audio import Pipeline, Audio
from pyannote.core import Segment
import whisper
from diarize_post_sr import convert_to_wav


WHISPERMODEL = "tiny.en"
TOKEN_FILE = "HF_TOK.txt"
audio_file = "../scratch/audio/earnings21-4341191.mp3"
output_dir = Path("../scratch/transcripts/tiny.en-diarize_pre_sr") / Path(audio_file).name
diarization_result_file = output_dir / f"{Path(audio_file).stem}-diarization.json"
final_reco_file = output_dir / f"{Path(audio_file).stem}-{output_dir.parent.name}.json"


os.makedirs(output_dir, exist_ok=True)


print("Creating pyannote diarization pipeline ..")
with open(TOKEN_FILE) as f:
    hf_tok = f.read().strip()
pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization@2.1", use_auth_token=hf_tok)
pipeline.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

# run pyannote diarization pipeline and save resulting segments
audio_file, _ = convert_to_wav(audio_file)
if not Path(diarization_result_file).is_file():
    print("Processing audio with pyannote diarization pipeline ..")
    diarization = pipeline(audio_file)
    diarized_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        s = turn.for_json()
        s['speaker'] = speaker
        diarized_segments.append(s)
    with open(diarization_result_file, 'w') as f:
        json.dump(diarized_segments, f, indent=4)
else:
    with open(diarization_result_file) as f:
        diarized_segments = json.load(f)

# summarize diarization result
segment_durations = [ s['end']-s['start'] for s in diarized_segments ]
num_speakers = len(set([s['speaker'] for s in diarized_segments]))
print(f"Diarized into {len(segment_durations)} segments with min/median/max duration: \
    {min(segment_durations):.2f}/{median(segment_durations):.2f}/{max(segment_durations):.2f}")
print(f"Detected {num_speakers} unique speakers")

# # visualize / explore diarization result
# from pyannote.core import notebook
# from pyannote.core import Segment
# # notebook.crop = Segment(0.0, 600.0)  # zoom into a region
# notebook.crop = Segment(3000.0, 3600.0)  # zoom into a region
# # notebook.reset()
# diarization

# transcribe cropped segments with whisper
model = whisper.load_model(WHISPERMODEL)

pn_audio = Audio(sample_rate=16_000, mono=True)
print(f"Transcribing file with whisper ..")
segments = []
for segment in tqdm(diarized_segments):
    waveform, sr = pn_audio.crop(audio_file, Segment(segment['start'], segment['end']))
    s = model.transcribe(waveform.squeeze().numpy(), verbose=None)['segments']
    for si in s:
        si['start'] += segment['start']
        si['end'] += segment['end']
        si['speaker'] = segment['speaker']
    segments.extend(s)

reco = dict(segments=segments)
reco["text"] = " ".join([s["text"] for s in reco["segments"]])
with open(final_reco_file, 'w') as fp:
    json.dump(reco, fp)
