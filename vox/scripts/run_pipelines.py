DESCRIPTION = """
Script that will run the following pipelines:
1. transcribe audio with whisper 
2. apply post_sr diarization
3. run pre_sr diarization and retranscribe audio
4. score all the results 
"""

import os
import sys
import json
import shutil
import logging
import pandas as pd
from pathlib import Path
from diarize_post_sr import add_speakers_to_segments
from diarize_pre_sr import run_pre_sr_pipeline

sys.path.append(str(Path(__file__).parent.parent))  # vox directory
from score import score_fstalign

sys.path.append(
    str(Path(__file__).parent.parent.parent)
)  # root directory of repo, above vox
import whisper
import whisper.utils as wutils


# set up logging with timestamp in 24hr format
logging.basicConfig(
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)

WHISPERMODEL = "tiny.en"


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
        help="pipelines to run",
        default="all",
        choices=["all", "1.4", "1"],
    )
    args = parser.parse_args()

    files_to_score = []
    pipelines_to_run = (
        args.pipelines_to_run.split(".")
        if args.pipelines_to_run != "all"
        else ["1", "2", "3", "4"]
    )

    # 1. transcribe audio with whisper
    logger.info("Transcribing audio with whisper ..")
    audio_file = args.audio_file
    output_dir = args.output_dir
    ref_file = Path(args.ref_file).resolve()
    os.makedirs(output_dir, exist_ok=True)
    transcribe_result_file = (
        Path(output_dir) / f"{Path(audio_file).name}.json"
    ).resolve()
    files_to_score.append((1, transcribe_result_file))

    if not Path(transcribe_result_file).is_file():
        model = whisper.load_model(args.whisper_model)
        result = model.transcribe(
            audio_file, verbose=False, condition_on_previous_text=True, beam_size=4
        )
        writer = wutils.get_writer("all", output_dir)
        writer(result, audio_file)
    else:
        with open(transcribe_result_file) as f:
            result = json.load(f)

    # 2. apply post_sr diarization
    if "2" in pipelines_to_run:
        logger.info("Applying post_sr diarization ..")
        drz_post_sr_output_dir = args.output_dir + "_drz_post_sr"
        os.makedirs(drz_post_sr_output_dir, exist_ok=True)
        drz_post_sr_reco_file = (
            Path(drz_post_sr_output_dir) / f"{Path(audio_file).name}.json"
        ).resolve()
        files_to_score.append((2, drz_post_sr_reco_file))

        if not Path(drz_post_sr_reco_file).is_file():
            result["segments"] = add_speakers_to_segments(
                audio_file, result["segments"], num_speakers=int(args.num_speakers)
            )
            writer = wutils.get_writer("all", drz_post_sr_output_dir)
            writer(result, audio_file)

    # 3. run pre_sr diarization and retranscribe audio
    if "3" in pipelines_to_run:
        logger.info("Running pre_sr diarization and retranscribing segmented audio ..")
        drz_pre_sr_output_dir = args.output_dir + "_drz_pre_sr"
        os.makedirs(drz_pre_sr_output_dir, exist_ok=True)
        drz_pre_sr_reco_file = (
            Path(drz_pre_sr_output_dir) / f"{Path(audio_file).name}.json"
        ).resolve()
        files_to_score.append((3, drz_pre_sr_reco_file))

        if not Path(drz_pre_sr_reco_file).is_file():
            result = run_pre_sr_pipeline(audio_file, drz_pre_sr_output_dir)
            writer = wutils.get_writer("all", drz_pre_sr_output_dir)
            writer(result, audio_file)

    # 4. score all the results
    if "4" in pipelines_to_run:
        logger.info("Scoring all the results ..")
        results = []
        os.chdir(Path(__file__).parent.parent)  # change to vox parent directory
        for i, reco_file in files_to_score:
            modes = ["segment", "punctuation"] if i == 1 else ["segment"]
            if args.pipelines_to_run == "1.4":
                modes.append("token")
            for mode in modes:
                # convert reco_file to result_name in this way
                # e.g. /home/whisper/vox/tiny.en/d1/f.json -> tiny.en-d1
                result_name = "__".join(Path(reco_file).parts[-3:-1])
                logger.info(f"Scoring {result_name} with mode {mode} ..")
                result, _ = score_fstalign(
                    ref_file, reco_file, result_name, speaker_turn_mode=mode
                )
                # result["method"] = f"spkturn_{mode}" if i == 1 else f"drz_post_sr" if i == 2 else f"drz_pre_sr"
                results.append(result)
        os.chdir(Path(__file__).parent)  # change back to vox/scripts directory

        results_df = pd.concat(results)
        results_df["audio_file"] = Path(audio_file).name
        print(results_df)

        # save results to tsv
        results_file = Path(output_dir) / "scoring_results.tsv"
        results_df.to_csv(results_file, sep="\t", index=False)
        logger.info(f"Saved results to {results_file}")
