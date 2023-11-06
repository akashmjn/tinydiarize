# tinydiarize 🐥🗣️

- *Speaker diarization* labels who said what in a transcript (e.g. Speaker A, Speaker B …). It is essential for conversation transcripts like meetings or podcasts.
- *tinydiarize*  aims to be a minimal, interpretable  extension of OpenAI's [Whisper](https://github.com/openai/whisper) models that adds speaker diarization with few extra dependencies (inspired by [minGPT](https://github.com/karpathy/minGPT)).
- This uses a finetuned model that adds special tokens to mark speaker changes [[1,2,3,4]](#references). It can use *both voice and semantic context to tell speakers apart*, which is a unique benefit of this approach.
- You can refer to [tdrz_dev](https://github.com/akashmjn/tinydiarize/tree/main/tdrz_dev) for a detailed analysis of performance. Note that this is intended to be a prototype/proof-of-concept.
- Experimental support is also added to [whisper.cpp](https://github.com/ggerganov/whisper.cpp#speaker-segmentation-via-tinydiarize-experimental) so this can run on consumer hardware like MacBooks and iPhones. A tiny change is needed to original inference code (<50 lines), enabling simple and cheap speaker segmentation, compared with conventional approaches.


## Demo

https://user-images.githubusercontent.com/13268767/229617067-eca0f614-d334-480d-9801-7c30d88acdc6.mp4

You can try it out on other such gems from YouTube using this notebook. [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akashmjn/tinydiarize/blob/main/notebooks/Demo_YouTube.ipynb)


## Quickstart 

Install `ffmpeg` following the [original repo](https://github.com/openai/whisper#Setup), then run:

```
pip install -e .
whisper --model small.en-tdrz AUDIO 
```

The only change is the `small.en-tdrz` model instead of `small.en`. That's it! 🎉


## What's included?

- Finetuned checkpoint for the `small.en-tdrz` model (located [here](whisper/__init__.py)) and example inference code (relevant edits in [[#4]](https://github.com/akashmjn/tinydiarize/pull/4) [[#11]](https://github.com/akashmjn/tinydiarize/pull/11)). This has the same dependencies as the original whisper repo.
- Tools for comparison and analysis (under [/tdrz_dev](tdrz_dev)):
    - A scoring tool to measure and compare accuracy on your own data in an easy to interpret way.
    - A reference script to run and compare various diarization pipelines.
    - A Jupyter notebook to compare and understand performance in detail.
- See [Roadmap](#roadmap) for more info.

We aim to demonstrate a starting point enabling anyone (or even OpenAI themselves!) to improve performance and extend support (multilingual, speech translation etc.).

## Performance

|metric|small.en|small.en-tdrz|
|:----|:----|:----|
|spk_turn_precision|-|97.7|
|spk_turn_recall|-|70.8|
|wer_overall|11.0|10.3|
|wer_speaker_switch|15.0|15.5|

On a (tiny) benchmark set of 3 [earnings calls](https://github.com/revdotcom/speech-datasets/tree/main/earnings21), `tdrz` gets near-perfect speaker turn precision at fairly decent recall. A similar WER is retained as the original model. Not too shabby for a tiny finetuning setup, and <10% extra inference cost!

Refer to [tdrz_dev](tdrz_dev/) for details on performance analysis and comparisons.

## More info
- Whisper `small.en` checkpoints were finetuned on ~100hrs of [AMI meetings](https://groups.inf.ed.ac.uk/ami/corpus/) using HuggingFace [Transformers](https://github.com/huggingface/transformers) and [Datasets](https://github.com/huggingface/datasets).
- With some tricks, this could be done relatively cheaply with just 30mins of 1 GPU training starting to produce decent results. Tiny indeed 😊.
- We used helpful tools from [pyannote](https://github.com/pyannote/pyannote-core) (the OG open-source diarization toolkit) for finetuning data preparation and also analyze its performance.
- We make use of the excellent open-source [revdotcom/fstalign](https://github.com/revdotcom/fstalign) tool for scoring and analysis.

## Gotchas

Note that this still an early proof-of-concept and there are a few things to be aware of:
- Only the `small.en` English model has been finetuned.
- Word-error-rate (WER) is close to original models, although not yet extensively tested. Ad-hoc inspection does show some differences in timestamp behavior (longer segments) or deletion errors. See the notebook under [tdrz_dev](tdrz_dev/) for details.
- Given a pretty tiny finetuning setup, there's likely a lot of room for further accuracy improvements.
- Only local diarization (segmentation into speaker turns) is handled so far. Extension with global diarization (speaker clustering) is planned for later.
- Stuff is still hacky and subject to change, so hold your horses just yet! 🐎

## Roadmap
- [x] inference code & demo
- [x] scoring and analysis tools
- [x] [whisper.cpp integration](https://github.com/ggerganov/whisper.cpp/pull/1058)
- [ ] *reproducible dataprep + finetuning\**
- [ ] *blog post explainer\**
- [ ] HuggingFace integration
- [ ] better LoRa-based `small.en` checkpoint
- [ ] possibly clustering with [NME-SC](https://github.com/tango4j/Auto-Tuning-Spectral-Clustering)?
- [ ] possibly `large-v2` checkpoint?

*\* is a pointer to the current state of the repo. Please see https://github.com/akashmjn/tinydiarize/issues/14 for an update on plans. TLDR; things have had to be put on pause :/*

## References

[[1]](https://arxiv.org/abs/1907.05337) Joint Speech Recognition and Speaker Diarization via Sequence Transduction
[[2]](https://arxiv.org/abs/2003.12687) Serialized Output Training for End-to-End Overlapped Speech Recognition
[[3]](https://arxiv.org/abs/2109.11641) Turn-to-Diarize: Online Speaker Diarization Constrained by Transformer Transducer Speaker Turn Detection
[[4]](https://arxiv.org/abs/2305.18747) Adapting Multi-Lingual ASR Models for Handling Multiple Talkers

For information on the underlying Whisper model, please refer to the [original documentation (release: `20230308`)](https://github.com/openai/whisper/tree/v20230308)

## License

Code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.

## Citation

If you please to use this in your research, you can cite this work as 
```
@software{mahajan2023tinydiarize,
  author = {Mahajan, Akash},
  month = {08},
  title = {tinydiarize: Minimal extension of Whisper for speaker segmentation with special tokens},
  url = {https://github.com/akashmjn/tinyDiarize},
  year = {2023}
}
```

