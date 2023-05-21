import json
import logging
import os
import sys
from pathlib import Path
from statistics import median

import torch
from diarize_post_sr import convert_to_wav
from pyannote.audio import Audio, Pipeline
from pyannote.core import Segment
from tqdm import tqdm

sys.path.append(
    str(Path(__file__).parent.parent.parent)
)  # root directory of repo, above tinydiarize
import whisper  # noqa: E402
import whisper.utils as wutils  # noqa: E402

WHISPERMODEL = "tiny.en"
TOKEN_FILE = "HF_TOK.txt"

logger = logging.getLogger(__name__)


def run_pyannote_pipeline(audio_file, num_speakers=None):
    # run pyannote diarization pipeline and save resulting segments
    audio_file, _ = convert_to_wav(audio_file)
    logging.info("Creating pyannote diarization pipeline ..")
    with open(TOKEN_FILE) as f:
        hf_tok = f.read().strip()
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization@2.1", use_auth_token=hf_tok
    )
    pipeline.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))

    logging.info("Processing audio with pyannote diarization pipeline ..")
    diarization = pipeline(audio_file, num_speakers=num_speakers)
    diarized_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        s = {"start": turn.start, "end": turn.end}
        s["speaker"] = speaker
        diarized_segments.append(s)

    return diarized_segments


def transcribe_cropped_segments(audio_file, diarized_segments):
    # transcribe cropped segments with whisper
    model = whisper.load_model(WHISPERMODEL)
    pn_audio = Audio(sample_rate=16_000, mono=True)
    logging.info("Transcribing file with whisper ..")
    result = dict(text="", segments=[])
    for segment in tqdm(diarized_segments):
        waveform, sr = pn_audio.crop(
            audio_file, Segment(segment["start"], segment["end"])
        )
        r = model.transcribe(waveform.squeeze().numpy(), verbose=None)
        # make all sub-segments relative to the original audio file
        for si in r["segments"]:
            si["start"] += segment["start"]
            si["end"] += segment["start"]
            si["speaker"] = segment["speaker"]

        result["text"] += r["text"] + " "
        result["segments"].extend(r["segments"])

    return result


def run_pre_sr_pipeline(audio_file, output_dir, num_speakers=None):
    os.makedirs(output_dir, exist_ok=True)
    audio_file, _ = convert_to_wav(audio_file)

    # TODO@Akash - wrap this pattern into a decorator
    diarization_result_file = (
        Path(output_dir) / f"{Path(audio_file).stem}-diarization.json"
    )
    if not Path(diarization_result_file).is_file():
        diarized_segments = run_pyannote_pipeline(audio_file, num_speakers)
        with open(diarization_result_file, "w") as f:
            json.dump(diarized_segments, f, indent=4)
    else:
        with open(diarization_result_file) as f:
            diarized_segments = json.load(f)

    # summarize diarization result
    segment_durations = [s["end"] - s["start"] for s in diarized_segments]
    num_speakers = len(set([s["speaker"] for s in diarized_segments]))
    logging.info(
        f"Diarized into {len(segment_durations)} segments with min/median/max duration: \
        {min(segment_durations):.2f}/{median(segment_durations):.2f}/{max(segment_durations):.2f}"
    )
    logging.info(f"Detected {num_speakers} unique speakers")

    # # visualize / explore diarization result
    # from pyannote.core import notebook
    # from pyannote.core import Segment
    # # notebook.crop = Segment(0.0, 600.0)  # zoom into a region
    # notebook.crop = Segment(3000.0, 3600.0)  # zoom into a region
    # # notebook.reset()

    # TODO@Akash - wrap this pattern into a decorator
    final_reco_file = Path(output_dir) / f"{Path(audio_file).stem}.json"
    if not Path(final_reco_file).is_file():
        result = transcribe_cropped_segments(audio_file, diarized_segments)
        # with open(final_reco_file, 'w') as f:
        # json.dump(result, f)
    else:
        with open(final_reco_file) as f:
            result = json.load(f)

    return result


if __name__ == "__main__":
    # get arguments from command line
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio", type=str, default="../scratch/audio/earnings21-4341191.mp3"
    )
    parser.add_argument(
        "output_dir",
        type=str,
        default="../scratch/transcripts/tiny.en_drzpresr/earnings21-4341191/",
    )
    parser.add_argument("--num_speakers", type=int, default=None)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    # run pipeline
    result = run_pre_sr_pipeline(args.audio, args.output_dir, args.num_speakers)

    # save result
    writer = wutils.get_writer("all", args.output_dir)
    writer(result, args.audio)
