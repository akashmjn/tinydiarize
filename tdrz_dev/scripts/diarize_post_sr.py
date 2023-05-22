# Script adapted from https://huggingface.co/spaces/dwarkesh/whisper-speaker-recognition
import contextlib
import datetime
import json
import logging
import os
import subprocess
import wave
from pathlib import Path

import numpy as np
import torch
from pyannote.audio import Audio
from pyannote.audio.pipelines.speaker_verification import PretrainedSpeakerEmbedding
from pyannote.core import Segment
from sklearn.cluster import AgglomerativeClustering
from tqdm import tqdm

WHISPERMODEL = "tiny.en"

pyannote_audio = Audio()


def convert_to_wav(path):
    if path[-3:] != "wav":
        wav_path = ".".join(path.split(".")[:-1]) + ".wav"
        try:
            subprocess.call(["ffmpeg", "-i", path, "-ar", "16000", wav_path, "-y"])
        except Exception:
            return path, "Error: Could not convert file to .wav"
        path = wav_path
    return path, None


def get_duration(wav_path):
    with contextlib.closing(wave.open(wav_path, "r")) as f:
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
    assert (
        sample_rate == 16000
    ), f"Invalid sampling rate for spk embedding model {sample_rate}"
    return embedding_model(waveform[None])


def add_speaker_labels(segments, embeddings, num_speakers):
    clustering = AgglomerativeClustering(num_speakers).fit(embeddings)
    labels = clustering.labels_
    for i in range(len(segments)):
        segments[i]["speaker"] = "SPEAKER " + str(labels[i] + 1)


def time(secs):
    return datetime.timedelta(seconds=round(secs))


def get_output(segments):
    output = ""
    for i, segment in enumerate(segments):
        if i == 0 or segments[i - 1]["speaker"] != segment["speaker"]:
            if i != 0:
                output += "\n\n"
            output += segment["speaker"] + " " + str(time(segment["start"])) + "\n\n"
        output += segment["text"][1:] + " "
    return output


def add_speakers_to_segments(audio, segments, num_speakers):
    embedding_model = PretrainedSpeakerEmbedding(
        "speechbrain/spkrec-ecapa-voxceleb",
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    wav_path, error = convert_to_wav(audio)
    if error is not None:
        return error

    duration = get_duration(wav_path)
    if duration > 4 * 60 * 60:
        return "Audio duration too long"

    num_speakers = min(max(round(num_speakers), 1), len(segments))
    if len(segments) == 1:
        segments[0]["speaker"] = "SPEAKER 1"
    else:
        logging.info(f"Creating embeddings for {len(segments)} segments ..")
        embeddings = make_embeddings(embedding_model, wav_path, segments, duration)
        logging.info(f"Clustering embeddings into {num_speakers} speakers ..")
        add_speaker_labels(segments, embeddings, num_speakers)

    return segments


if __name__ == "__main__":
    # get arguments from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio", type=str, default="../scratch/audio/earnings21-4341191.mp3"
    )
    parser.add_argument(
        "reco_file",
        type=str,
        default="../scratch/transcripts/tiny.en-diarize_postsr/earnings21-4341191/earnings21-4341191-tiny.en.json",
    )
    parser.add_argument("--num_speakers", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    output_dir = (
        Path(args.reco_file).parent
        if args.output_dir is None
        else Path(args.output_dir)
    )
    os.makedirs(output_dir, exist_ok=True)

    print("Loading reco file ..")
    print(args.reco_file)
    with open(args.reco_file) as fp:
        reco = json.load(fp)

    print(f"Clustering segments into {args.num_speakers} speakers ..")
    segments = add_speakers_to_segments(args.audio, reco["segments"], args.num_speakers)

    output_file = output_dir / Path(args.reco_file).name.replace(
        ".json", f"_drzpostsr_{args.num_speakers}.json"
    )
    with open(output_file, "w") as fp:
        json.dump(reco, fp)

    print("Written to output file ..")
    print(output_file)
