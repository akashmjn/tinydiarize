# tinydiarize dev 🔨📊

This directory contains tools to aid development and analysis. It has extra [dependencies](#setup) that are not required for inference. Contents:
- [score.py](score.py) to measure and compare accuracy on your own data with easy to interpret metrics (WER, speaker turn precision/recall).
- [run_pipelines.py](scripts/run_pipelines.py) shows how to run and compare various diarization pipelines.
- [analysis.ipynb](notebooks/analysis.ipynb) walks through a comparison of various pipelines with a deep dive to understand sources of errors.
- Code to reproduce finetuning will also be released shortly.

This can be used to reproduce results and take a closer look at analysis from the blog post.

## Analysis 

In the accompanying notebook [analysis.ipynb](https://nbviewer.org/github/akashmjn/tinydiarize/blob/proofread-1/tdrz_dev/notebooks/analysis.ipynb) ([![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akashmjn/tinydiarize/blob/main/tdrz_dev/notebooks/analysis.ipynb)) we show that:
- Whisper models already have a good internal representation of speaker turns via both acoustic and semantic cues (much like a punctuation).
- Their placement of `punctuation` tokens appears to be very consistent with speaker turns (high recall).
- Whisper-produced time segments `segment_timestamped` (common workaround used [here](https://huggingface.co/spaces/vumichien/Whisper_speaker_diarization)) are not as consistent.
- Conventional acoustic-embedding diarization approaches like `pyannote_pre_sr` have good all-round performance, but have a gap with the best possible precision or recall. This is due to a design limitation that causes trouble with short segments & quick speaker turns.
- The `tdrz_token` prototype shows that we can cheaply isolate Whisper's speaker representations with a simple finetuning setup, without too much impact on word error rate.
- With improvments to the finetuning setup, we can expect to see strong performance from `tdrz_token` as it can use both voice and semantic context to tell speakers apart, which is a unique benefit of this approach.

|model|small.en| | |small.en-tdrz|
|:----|:----|:----|:----|:----|
|method|punctuation|pyannote_pre_sr|segment_timestamped|tdrz_token|
|metric| | | | |
|spk_turn_precision|19.49|83.39|14.52|98.16|
|spk_turn_recall|92.03|78.41|86.71|70.76|
|wer_overall|10.95|12.94|10.95|10.30|
|wer_speaker_switch|14.96|23.13|14.96|15.56|

![metrics](barplots.png)

## Runtime cost estimate

These numbers were tested using the earnings21-4374910 call (33.8 min) on a Quadro RTX 5000 GPU. Whisper is run with beam_size=4 and condition_on_previous_text=True.

Local diarization done by `tdrz` comes at a marginal added cost. If we account for an additional clustering step (implemented [here](scripts/diarize_post_sr.py)), this can still be quite cheap overall.

|Stage|Runtime (s)|Extra cost (%)|
|:----|:----|:----|
|Whisper.transcribe|121.2|-|
|Pyannote diarization|56.6|47%|
|Clustering whisper time segments|3.4|3%|
|Whisper.transcribe (tdrz)|131.5|8%|


## Setup

You'll need to setup the following dependencies.

1. In a fresh python environment (I used python=3.9), run the following (from the root of the github repo):
```
    pip install -e .                       
    cd tinydiarize
    pip install -r extra_requirements.txt
```
2. `docker pull revdotcom/fstalign`. This installs [revdotcom/fstalign](https://github.com/revdotcom/fstalign) for scoring
3. [Pyannote](https://github.com/pyannote/pyannote-audio) consent prerequisites:
    - visit hf.co/pyannote/speaker-diarization and accept user conditions
    - visit hf.co/pyannote/segmentation and accept user conditions
    - visit hf.co/settings/tokens to create an access token and save it to a text file at `tdrz_dev/scripts/HF_TOK.txt`.
