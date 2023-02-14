# Script adapted from https://huggingface.co/spaces/dwarkesh/whisper-speaker-recognition

import datetime
import subprocess
import torch
import wave
import contextlib
import json
import sys
import numpy as np
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.audio import Audio
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm
from pathlib import Path
import whisper

WHISPERMODEL = "tiny.en"


pyannote_audio = Audio()

def convert_to_wav(path):
  if path[-3:] != 'wav':
    wav_path = '.'.join(path.split('.')[:-1]) + '.wav'
    try:
      subprocess.call(['ffmpeg', '-i', path, '-ar', '16000', wav_path, '-y'])
    except:
      return path, 'Error: Could not convert file to .wav'
    path = wav_path
  return path, None

def get_duration(wav_path):
  with contextlib.closing(wave.open(wav_path,'r')) as f:
    frames = f.getnframes()
    rate = f.getframerate()
    return frames / float(rate)

def make_embeddings(embedding_model, wav_path, segments, duration):
  embeddings = np.zeros(shape=(len(segments), 192))
  for i, segment in enumerate(tqdm(segments)):
    embeddings[i] = segment_embedding(embedding_model, wav_path, segment, duration)
  return np.nan_to_num(embeddings)

def segment_embedding(embedding_model, wav_path, segment, duration):
  start = segment["start"]
  # Whisper overshoots the end timestamp in the last segment
  end = min(duration, segment["end"])
  clip = Segment(start, end)
  waveform, sample_rate = pyannote_audio.crop(wav_path, clip)
  assert sample_rate==16000, f"Invalid sampling rate for spk embedding model {sample_rate}"
  return embedding_model(waveform[None])

def add_speaker_labels(segments, embeddings, num_speakers):
  clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
  labels = clustering.labels_
  for i in range(len(segments)):
    segments[i]["speaker"] = 'SPEAKER ' + str(labels[i] + 1)

def time(secs):
  return datetime.timedelta(seconds=round(secs))

def get_output(segments):
  output = ''
  for (i, segment) in enumerate(segments):
    if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
      if i != 0:
        output += '\n\n'
      output += segment["speaker"] + ' ' + str(time(segment["start"])) + '\n\n'
    output += segment["text"][1:] + ' '
  return output

def transcribe(audio, output_file):
  model = whisper.load_model(WHISPERMODEL)
  print(f"Transcribing file with whisper ..")
  result = model.transcribe(audio, verbose=True)
  segments = result["segments"]

  with open(output_file, 'w') as fp:
    whisper.utils.write_json(segments, fp)
  
  return segments

def add_speakers_to_segments(audio, reco, num_speakers, output_file):
  segments = reco["segments"]
  embedding_model = PretrainedSpeakerEmbedding( 
      "speechbrain/spkrec-ecapa-voxceleb",
      device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  )

  # TODO@Akash - work out proper sample rate
  wav_path, error = convert_to_wav(audio)
  if error is not None:
    return error

  duration = get_duration(wav_path)
  if duration > 4 * 60 * 60:
    return "Audio duration too long"

  num_speakers = min(max(round(num_speakers), 1), len(segments))
  if len(segments) == 1:
    segments[0]['speaker'] = 'SPEAKER 1'
  else:
    print(f"Creating embeddings for {len(segments)} segments ..")
    embeddings = make_embeddings(embedding_model, wav_path, segments, duration)
    print(f"Clustering embeddings into {num_speakers} speakers ..")
    add_speaker_labels(segments, embeddings, num_speakers)

  with open(output_file, 'w') as fp:
    json.dump(reco, fp)
  
  return segments

# def transcribe_with_speakers(audio, num_speakers):
#   pass


if __name__ == "__main__":

  audio = "../scratch/audio/earnings21-4341191.mp3"
  reco_file = "../scratch/transcripts/tiny.en-diarize_postsr/earnings21-4341191/earnings21-4341191-tiny.en.json"
  num_speakers = 5 if sys.argv[1] is None else int(sys.argv[1])
  output_dir = Path(reco_file).parent

  # segments = transcribe(audio, reco_file)
  print("Loading reco file ..")
  print(reco_file)
  with open(reco_file) as fp:
    reco = json.load(fp)

  output_file = reco_file.replace(".json", f"_drzpostsr_{num_speakers}.json")
  print(f"Clustering segments into {num_speakers} speakers ..")
  segments = add_speakers_to_segments(audio, reco, num_speakers, output_file)

  print("Written to output file ..")
  print(output_file)
