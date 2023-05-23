import json
import logging
import os
import sys
from pathlib import Path

import pandas as pd
from diarize_post_sr import add_speakers_to_segments
from diarize_pre_sr import run_pre_sr_pipeline

import whisper
import whisper.utils as wutils

sys.path.append(str(Path(__file__).resolve().parent.parent))
from score import score_fstalign  # noqa: E402

DESCRIPTION = """
Script that will run the following pipelines:
1. transcribe audio with whisper
2. apply post_sr diarization by clustering whisper segments
3. run pyannote pre_sr diarization and retranscribe segmented audio
4. transcribe audio with whisper-tdrz
5. score all the results
"""


def setup_logging(output_dir, audio_name):
    log_dir = f"{output_dir}/run_pipeline_logs"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s %(levelname)s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        level=logging.DEBUG,
    )
    # setup logging to also write to file in output_dir with same format as console
    fh = logging.FileHandler(f"{log_dir}/run_pipelines-{audio_name}.log")
    fh.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh.setFormatter(formatter)
    logging.getLogger().addHandler(fh)
    return log_dir


WHISPERMODEL = "small.en"
TOKEN_FILE = "HF_TOK.txt"


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("audio_file", help="path to audio file")
    parser.add_argument(
        "ref_file", help="path to reference transcript file in nlp format"
    )
    parser.add_argument("output_dir", help="path to output directory")
    parser.add_argument(
        "--num_speakers",
        type=int,
        help="provide the oracle number as we are only evaluating local diarization",
    )
    parser.add_argument(
        "--pipelines_to_run",
        help="pipelines to run. either 'all' or a comma separated list of numbers 1-5",
        default="all",
    )
    parser.add_argument(
        "--whisper_model",
        help="valid whisper model name or path to checkpoint",
        default=WHISPERMODEL,
    )
    parser.add_argument(
        "--hf_token_file",
        help="text file containing HuggingFace token required for pyannote",
        default=TOKEN_FILE,
    )
    args = parser.parse_args()

    files_to_score = []
    pipelines_to_run = (
        args.pipelines_to_run.split(",")
        if args.pipelines_to_run != "all"
        else ["1", "2", "3", "4", "5"]
    )

    audio_file, audio_name = args.audio_file, Path(args.audio_file).stem
    log_dir = setup_logging(args.output_dir, audio_name)
    reco_fname = f"{audio_name}.json"
    ref_file = Path(args.ref_file).resolve()
    whisper_output_dir = f"{args.output_dir}/{args.whisper_model}/{audio_name}"

    # 1. transcribe audio with whisper
    if "1" in pipelines_to_run:
        logging.info("Transcribing audio with whisper ..")
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
        logging.info("Applying post_sr diarization ..")
        drz_post_sr_output_dir = whisper_output_dir + "_drz_post_sr"
        os.makedirs(drz_post_sr_output_dir, exist_ok=True)
        drz_post_sr_reco_file = (Path(drz_post_sr_output_dir) / reco_fname).resolve()
        if not Path(drz_post_sr_reco_file).is_file():
            result["segments"] = add_speakers_to_segments(
                audio_file, result["segments"], num_speakers=args.num_speakers
            )
            writer = wutils.get_writer("all", drz_post_sr_output_dir)
            writer(result, audio_file)

        files_to_score.append((drz_post_sr_reco_file, "segment"))

    # 3. run pyannote pre_sr diarization and retranscribe segmented audio
    if "3" in pipelines_to_run:
        logging.info("Running pre_sr diarization and retranscribing segmented audio ..")
        drz_pre_sr_output_dir = whisper_output_dir + "_drz_pre_sr"
        os.makedirs(drz_pre_sr_output_dir, exist_ok=True)
        drz_pre_sr_reco_file = (Path(drz_pre_sr_output_dir) / reco_fname).resolve()
        if not Path(drz_pre_sr_reco_file).is_file():
            result = run_pre_sr_pipeline(
                audio_file,
                drz_pre_sr_output_dir,
                num_speakers=args.num_speakers,
                hf_token_file=args.hf_token_file,
                whisper_model=args.whisper_model,
            )
            writer = wutils.get_writer("all", drz_pre_sr_output_dir)
            writer(result, audio_file)

        files_to_score.append((drz_pre_sr_reco_file, "segment"))

    # 4. transcribe audio with whisper-tdrz
    if "4" in pipelines_to_run:
        logging.info("Transcribing audio with whisper tinydiarize..")
        tdrz_output_dir = f"{args.output_dir}/{args.whisper_model}-tdrz/{audio_name}"
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
        logging.info("Scoring all the results ..")
        results = []

        os.chdir(Path(__file__).parent.parent)  # change to tdrz_dev parent directory
        for reco_file, scoring_mode in files_to_score:
            # convert reco_file to result_name in this way
            # e.g. /home/whisper/tdrz_dev/tiny.en/d1/f.json -> tiny.en-d1
            result_name = "__".join(Path(reco_file).parts[-3:-1])
            logging.info(f"Scoring {result_name} with mode {scoring_mode} ..")
            result, _ = score_fstalign(
                ref_file, reco_file, result_name, speaker_turn_mode=scoring_mode
            )
            results.append(result)
        os.chdir(Path(__file__).parent)  # change back to tdrz_dev/scripts directory

        results_df = pd.concat(results)
        results_df["audio_file"] = Path(audio_file).name
        print(results_df)

        # save results to tsv
        results_file = f"{log_dir}/scoring_results-{audio_name}.tsv"
        results_df.to_csv(results_file, sep="\t", index=False)
        logging.info(f"Saved results to {results_file}")
