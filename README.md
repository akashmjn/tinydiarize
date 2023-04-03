# tinyDiarize 🐥🗣️

This is a minimal extension of OpenAI's [Whisper](https://github.com/openai/whisper) models that adds speaker diarization to transcripts via special `<|speakerturn|>` tokens. It can be used as a drop-in replacement for `whisper.transcribe` with the same API, and no extra dependencies.

![demo](trim-tinydiarize.gif)

## Quickstart 

Simply run the original setup and use the `small.en-tdrz` model instead of `small.en`. That's it! 🎉

```
pip install -e .
whisper AUDIO --model small.en-tdrz SAME_CLI_ARGS
```

*(the code will auto-download the finetuned checkpoint, see  `whisper.__init__` for info)*

You can try it out on videos from YouTube using this notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akashmjn/tinyDiarize/blob/demo/notebooks/Demo_YouTube.ipynb)

https://user-images.githubusercontent.com/13268767/229617067-eca0f614-d334-480d-9801-7c30d88acdc6.mp4

## Why do this?

- *Speaker diarization* is the task of identifying who spoke when in an audio recording. Along with spoken content, it is a key part of creating who-spoke-what transcripts, such as those for podcasts.
- *tinyDiarize*  aims to be a minimal, interpretable  extension of original Whisper models (inspired by [minGPT](https://github.com/karpathy/minGPT)) that keeps extra dependencies to a minimum. 
- By extending models with special `<|speakerturn|>` tokens [[citations]](#references) a key part of the task can be solved cleanly, effectively, and at no extra cost. Stay tuned for details in an upcoming blog post! 📺
- The simplicity (same structure checkpoint, few line edits of inference code) has the added benefit of ease of integration into existing ports like [whisper.cpp](https://github.com/ggerganov/whisper.cpp) that runs on MacBooks and iPhones.
- By also releasing reproducible finetuning, we hope to enable others (or even OpenAI themselves!) to improve performance and extend support (multilingual, speech translation etc.)

## More info 
- Whisper `small.en` checkpoints were finetuned using HuggingFace [Transformers](https://github.com/huggingface/transformers) and [Datasets](https://github.com/huggingface/datasets). This could be done relatively cheaply with just 30mins of 1 GPU training :)
- Code will be released shortly for full reproducibility.
- An important contribution of this repo is also a scoring/analysis setup using [revdotcom/fstalign](https://github.com/revdotcom/fstalign) that allows for interpretable error inspection and side-by-side analysis.
- A blog post and accompanying Jupyter notebooks will be released soon with more details.

![metrics](landing-page-metrics.png)

## Gotchas

Note that this is still WIP and there are a few things to be aware of:
- This was done only for the `small.en` English model mainly to demonstrate feasibility. 
- Initial tests show it's possible to have minimal impact on the original accuracy (WER) of models. Just putting this in gotchas here until a more thorough analysis is done.
- Only local diarization is handled so far (speaker turns). A (TBD) clustering step will be needed to group speaker turns into speaker A/B/C etc.
- Stuff is still quite hacky and subject to change, so bear with us until things are released! 🙏

## References

[[1]](https://arxiv.org/abs/1907.05337) Joint Speech Recognition and Speaker Diarization via Sequence Transduction
[[2]](https://arxiv.org/abs/2003.12687) Serialized Output Training for End-to-End Overlapped Speech Recognition
[[3]](https://arxiv.org/abs/2109.11641) Turn-to-Diarize: Online Speaker Diarization Constrained by Transformer Transducer Speaker Turn Detection

For information on the underlying Whisper model, please refer to the [original documentation (release: `20230308`)](https://github.com/openai/whisper/tree/v20230308)

## License

Code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.
