# Setup

This section contains code and notebooks to reproduce the results and dive deeper into the analysis in the blog post. Before you get started, you'll need to setup the following dependencies.

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
    - visit hf.co/settings/tokens to create an access token and save it to a text file at tdrz_dev/scripts/HF_TOK.txt

# Analysis 

Code to reproduce these results and analysis in detail is available in [notebooks/analysis.ipynb](notebooks/analysis.ipynb). Code for finetuning will be released shortly.

|model|small.en| | |small.en-tdrz|
|:----|:----|:----|:----|:----|
|method|punctuation|pyannote_pre_sr|segment_timestamped|tdrz_token|
|metric| | | | |
|spk_turn_precision|19.49|83.39|14.52|98.16|
|spk_turn_recall|92.03|78.41|86.71|70.76|
|wer_overall|10.95|12.94|10.95|10.30|
|wer_speaker_switch|14.96|23.13|14.96|15.56|

![metrics](barplots.png)

|Stage|Runtime (s)|Extra cost (%)|
|:----|:----|:----|
|Whisper.transcribe|121.2|-|
|Pyannote diarization|56.6|47%|
|Clustering whisper segments|3.4|3%|
|Whisper.transcribe (tdrz)|131.5|8%|
