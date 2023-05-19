import json
import logging
import os
from pathlib import Path

import pandas as pd
import whisper
import whisper.utils as wutils
from diarize_post_sr import add_speakers_to_segments
from diarize_pre_sr import run_pre_sr_pipeline
from tinydiarize.score import score_fstalign

DESCRIPTION = """
Script that will run the following pipelines:
1. transcribe audio with whisper
2. apply post_sr diarization by clustering whisper segments
3. run pyannote pre_sr diarization and retranscribe segmented audio
4. transcribe audio with whisper-tdrz
5. score all the results
"""

# set up logging with timestamp in 24hr format
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

WHISPERMODEL = "small.en"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("audio_file", help="path to audio file")
    parser.add_argument(
        "ref_file", help="path to reference transcript file in nlp format"
    )
    parser.add_argument("output_dir", help="path to output directory")
    parser.add_argument("--num_speakers", help="number of speakers")
    parser.add_argument(
        "--whisper_model",
        help="valid whisper model name or path to checkpoint",
        default=WHISPERMODEL,
    )
    parser.add_argument(
        "--pipelines_to_run",
        help="pipelines to run. either 'all' or a comma separated list of numbers 1-5",
        default="all",
    )
    args = parser.parse_args()

    files_to_score = []
    pipelines_to_run = (
        args.pipelines_to_run.split(",")
        if args.pipelines_to_run != "all"
        else ["1", "2", "3", "4", "5"]
    )

    audio_file = args.audio_file
    reco_fname = f"{Path(audio_file).stem}.json"
    ref_file = Path(args.ref_file).resolve()

    # 1. transcribe audio with whisper
    if "1" in pipelines_to_run:
        logger.info("Transcribing audio with whisper ..")
        whisper_output_dir = (
            f"{args.output_dir}/{args.whisper_model}/{Path(audio_file).stem}"
        )
        os.makedirs(whisper_output_dir, exist_ok=True)
        transcribe_result_file = (Path(whisper_output_dir) / reco_fname).resolve()
        if not Path(transcribe_result_file).is_file():
            model = whisper.load_model(args.whisper_model)
            result = model.transcribe(
                audio_file, verbose=False, condition_on_previous_text=True, beam_size=4
            )
            writer = wutils.get_writer("all", whisper_output_dir)
            writer(result, audio_file)
        else:
            with open(transcribe_result_file) as f:
                result = json.load(f)

        files_to_score.append((transcribe_result_file, "segment"))
        files_to_score.append((transcribe_result_file, "punctuation"))

    # 2. apply post_sr diarization by clustering whisper segments
    if "2" in pipelines_to_run:
        logger.info("Applying post_sr diarization ..")
        drz_post_sr_output_dir = whisper_output_dir + "_drz_post_sr"
        os.makedirs(drz_post_sr_output_dir, exist_ok=True)
        drz_post_sr_reco_file = (Path(drz_post_sr_output_dir) / reco_fname).resolve()
        if not Path(drz_post_sr_reco_file).is_file():
            result["segments"] = add_speakers_to_segments(
                audio_file, result["segments"], num_speakers=int(args.num_speakers)
            )
            writer = wutils.get_writer("all", drz_post_sr_output_dir)
            writer(result, audio_file)

        files_to_score.append((drz_post_sr_reco_file, "segment"))

    # 3. run pyannote pre_sr diarization and retranscribe segmented audio
    if "3" in pipelines_to_run:
        logger.info("Running pre_sr diarization and retranscribing segmented audio ..")
        drz_pre_sr_output_dir = whisper_output_dir + "_drz_pre_sr"
        os.makedirs(drz_pre_sr_output_dir, exist_ok=True)
        drz_pre_sr_reco_file = (Path(drz_pre_sr_output_dir) / reco_fname).resolve()
        if not Path(drz_pre_sr_reco_file).is_file():
            result = run_pre_sr_pipeline(audio_file, drz_pre_sr_output_dir)
            writer = wutils.get_writer("all", drz_pre_sr_output_dir)
            writer(result, audio_file)

        files_to_score.append((drz_pre_sr_reco_file, "segment"))

    # 4. transcribe audio with whisper-tdrz
    if "4" in pipelines_to_run:
        logger.info("Transcribing audio with whisper tinydiarize..")
        tdrz_output_dir = (
            f"{args.output_dir}/{args.whisper_model}-tdrz/{Path(audio_file).stem}"
        )
        os.makedirs(tdrz_output_dir, exist_ok=True)
        transcribe_result_file = (Path(tdrz_output_dir) / reco_fname).resolve()
        if not Path(transcribe_result_file).is_file():
            model = whisper.load_model(args.whisper_model + "-tdrz")
            result = model.transcribe(
                audio_file, verbose=False, condition_on_previous_text=True, beam_size=4
            )
            writer = wutils.get_writer("all", tdrz_output_dir)
            writer(result, audio_file)

        files_to_score.append((transcribe_result_file, "token"))
        files_to_score.append((transcribe_result_file, "segment"))
        files_to_score.append((transcribe_result_file, "punctuation"))

    # 5. score all the results
    if "5" in pipelines_to_run:
        logger.info("Scoring all the results ..")
        results = []

        os.chdir(Path(__file__).parent.parent)  # change to tinydiarize parent directory
        for reco_file, scoring_mode in files_to_score:
            # convert reco_file to result_name in this way
            # e.g. /home/whisper/tinydiarize/tiny.en/d1/f.json -> tiny.en-d1
            result_name = "__".join(Path(reco_file).parts[-3:-1])
            logger.info(f"Scoring {result_name} with mode {scoring_mode} ..")
            result, _ = score_fstalign(
                ref_file, reco_file, result_name, speaker_turn_mode=scoring_mode
            )
            results.append(result)
        os.chdir(Path(__file__).parent)  # change back to tinydiarize/scripts directory

        results_df = pd.concat(results)
        results_df["audio_file"] = Path(audio_file).name
        print(results_df)

        # save results to tsv
        results_file = Path(args.output_dir) / "scoring_results.tsv"
        results_df.to_csv(results_file, sep="\t", index=False)
        logger.info(f"Saved results to {results_file}")
