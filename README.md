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

## Demo

You can try it out on videos from YouTube using this notebook
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/akashmjn/tinyDiarize/blob/main/notebooks/Demo_YouTube.ipynb)

https://user-images.githubusercontent.com/13268767/229617067-eca0f614-d334-480d-9801-7c30d88acdc6.mp4

## What is this?

- *Speaker diarization* is the task of identifying who spoke when in an audio recording e.g. Speaker A... Speaker B... Along with spoken content, it is a key part of creating who-spoke-what transcripts, such as those for podcasts.
- *tinyDiarize*  aims to be a minimal, interpretable  extension of original Whisper models that keeps extra dependencies to a minimum (inspired by [minGPT](https://github.com/karpathy/minGPT)).
- It keeps the exact same Whisper model structure, with a <20 line change to the inference code [#4](https://github.com/akashmjn/tinyDiarize/pull/4). This makes it easy to integrate into existing ports like [whisper.cpp](https://github.com/ggerganov/whisper.cpp) that runs on consumer hardware like MacBooks and iPhones.
- It makes use of special `<|speakerturn|>` tokens [[reference]](#references) to tackle the task of local diarization cleanly, effectively, and at minimal extra inference cost. Stay tuned for details in an upcoming blog post! 📺
- By also making reproducible finetuning available, we hope provide a starting point for others (or even OpenAI themselves!) to improve performance and extend support (multilingual, speech translation etc.)

## More info 
- Whisper `small.en` checkpoints were finetuned on ~100hrs of [AMI meetings](https://groups.inf.ed.ac.uk/ami/corpus/) using HuggingFace [Transformers](https://github.com/huggingface/transformers) and [Datasets](https://github.com/huggingface/datasets). With some tricks, this could be done relatively cheaply with just 30mins of 1 GPU training 😊.
- Notably, we include scoring & analysis tools using [revdotcom/fstalign](https://github.com/revdotcom/fstalign) that allow for interpretable error inspection and side-by-side analysis.
- For a detailed look at results see [/tdrz_dev](/tdrz_dev/). It contains a detailed Jupyter notebook with code to reproduce results and compare \& inspect errors.
- Finetuning code will also be released shortly for full reproducibility.


## Gotchas

Note that this is still quite WIP and there are a few things to be aware of:
- As an initial proof-of-concept only the `small.en` English model has been finetuned.
- Preliminary tests show similar word-error-rate (WER) as original models, although this is on a very small setup. Ad-hoc inspection shows occasional significant differences in timestamp behavior (longer segments) or deletion errors.
- This will likely improve quite a lot with a less hacky finetuning setup.
- Only local diarization (segmentation into speaker turns) is handled so far. Extension with global diarization (speaker clustering) is planned for later.
- Stuff is still quite hacky and subject to change, so bear with us until things are stabilized! 🙏

## References

[[1]](https://arxiv.org/abs/1907.05337) Joint Speech Recognition and Speaker Diarization via Sequence Transduction
[[2]](https://arxiv.org/abs/2003.12687) Serialized Output Training for End-to-End Overlapped Speech Recognition
[[3]](https://arxiv.org/abs/2109.11641) Turn-to-Diarize: Online Speaker Diarization Constrained by Transformer Transducer Speaker Turn Detection

For information on the underlying Whisper model, please refer to the [original documentation (release: `20230308`)](https://github.com/openai/whisper/tree/v20230308)

## License

Code and model weights are released under the MIT License. See [LICENSE](https://github.com/openai/whisper/blob/main/LICENSE) for further details.
