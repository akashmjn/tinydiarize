import itertools

from transformers import WhisperTokenizer


class WhisperDiarizedTokenizer(WhisperTokenizer):
    """
    Wrapper tokenizer class that hacks original tokenizer to support
    (i) combining multiple segments with timestamps
    (ii) tokens for speaker turns

    # any other methods that need to be overridden? (see token_to_id and id_to_token)
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # self.set_prefix_tokens(language = "en", task = "transcribe", predict_timestamps = True)
        self.set_prefix_tokens(predict_timestamps=True)
        self.add_prefix_space = True

        self.timestamp_begin = 50363  # HACK
        self.speaker_turn_token = "<|speakerturn|>"
        # overriding unused <|startoflm|> token https://github.com/openai/whisper/discussions/414
        self.speaker_turn_token_id = 50359  # HACK
        self.nospeech_token = "<|nocaptions|>"
        self.nospeech_token_id = 50361  # HACK
        if ".en" not in self.name_or_path:  # multilingual model
            self.timestamp_begin += 1
            self.speaker_turn_token_id += 1
            self.nospeech_token_id += 1

    def timestamp_to_token(self, timestamp):
        timestamp = round(timestamp / 0.02) * 0.02  # discretize to 20ms
        token = int(timestamp / 0.02)
        if not 0 <= token <= 1500:
            print(f"WARNING: Timestamp {timestamp} is out of range")
            # clip to between 0 and 1500
            token = max(0, min(1500, token))
        return self.timestamp_begin + token

    def token_to_timestamp(self, token):
        return (token - self.timestamp_begin) * 0.02

    def _encode_segments(self, segments, add_speaker_turns=None):
        token_list = [*self.prefix_tokens]

        if len(segments) == 0:
            token_list.append(self.nospeech_token_id)
            token_list.append(self.eos_token_id)
            return token_list

        for i, segment in enumerate(segments):
            segment_tokens = []
            if self.predict_timestamps:
                segment_tokens.append(self.timestamp_to_token(segment["start"]))

            # prepend token to segment after speaker turn (previous segment has different speaker)
            if (
                add_speaker_turns == "after"
                and i > 0
                and segments[i - 1]["speaker"] != segment["speaker"]
            ):
                segment_tokens.append(self.speaker_turn_token_id)

            segment_tokens.extend(
                super().encode(segment["transcription"], add_special_tokens=False)
            )

            # append token to segment before speaker turn (next segment has different speaker)
            if (
                add_speaker_turns == "before"
                and i < len(segments) - 1
                and segments[i + 1]["speaker"] != segment["speaker"]
            ):
                segment_tokens.append(self.speaker_turn_token_id)

            # end time based on long form segmentation
            if segment.get("spliced", None) != "right" and self.predict_timestamps:
                segment_tokens.append(self.timestamp_to_token(segment["end"]))

            token_list.extend(segment_tokens)

        token_list.append(self.eos_token_id)
        return token_list

    def encode(self, *args, **kwargs):
        if isinstance(args[0], list):
            # TODO@Akash: add support for add_special_tokens like in original API
            return self._encode_segments(
                args[0], add_speaker_turns=kwargs.get("add_speaker_turns", "before")
            )
        else:
            return super().encode(*args, **kwargs)

    def _is_diarize_token(self, token):
        return token >= self.timestamp_begin or token == self.speaker_turn_token_id

    def _split_segment_tokens(self, tokens):
        # split a list into sub lists based on a condition
        result = [
            list(g)
            for k, g in itertools.groupby(tokens, lambda x: self._is_diarize_token(x))
        ]
        assert len(tokens) == sum([len(_) for _ in result])
        return result

    # TODO@Akash - make this match new order of speaker turn tokens and timestamps
    def decode_segments(self, tokens, *args, **kwargs):
        # decode a list of tokens into segments with timestamps and speaker turns
        segment_tokens = self._split_segment_tokens(tokens)
        result = []
        segment = dict(transcription="")
        for tokens in segment_tokens:
            # check if tokens starts with self.prefix_tokens
            if tokens[: len(self.prefix_tokens)] == self.prefix_tokens:
                segment["transcription"] += super().decode(tokens, *args, **kwargs)
                continue

            if self._is_diarize_token(tokens[0]):
                # can be either start; end; end,start; or end,speakerturn,start
                if len(tokens) == 1 and segment["transcription"] == "":
                    # current segment starts with it
                    segment["start"] = self.token_to_timestamp(tokens[0])
                elif len(tokens) == 1:
                    # current segment ends with it
                    segment["end"] = self.token_to_timestamp(tokens[0])
                else:
                    # end current segment
                    segment["end"] = self.token_to_timestamp(tokens[0])
                    result.append(segment)
                    # TODO@Akash: handle speaker turns
                    # start new segment
                    segment = dict(
                        transcription="", start=self.token_to_timestamp(tokens[-1])
                    )

            elif len(tokens) > 0:
                segment["transcription"] += super().decode(tokens, **kwargs)

        if segment["transcription"] != "":  # last segment
            result.append(segment)

        return result

    # TODO@Akash - make this match new order of speaker turn tokens and timestamps
    def decode(self, tokens, *args, **kwargs):
        decode_time_token = lambda token: f"<|{self.token_to_timestamp(token):.2f}|>"
        result = []
        segment_tokens = self._split_segment_tokens(tokens)
        for tokens in segment_tokens:
            if self._is_diarize_token(tokens[0]):
                for t in tokens:
                    if t == self.speaker_turn_token_id:
                        result.append(self.speaker_turn_token)
                    else:
                        result.append(decode_time_token(t))
            else:
                result.append(super().decode(tokens, **kwargs))
        return "".join(result)


def test_tokenizer(tokenizer, segments, add_speaker_turns="before"):
    # test that running encode + decode returns the same segments,
    # including timestamp & speaker turns
    print("\n", segments)
    tokens = tokenizer.encode(segments, add_speaker_turns=add_speaker_turns)
    print(tokens)
    decoded_back = tokenizer.decode_segments(tokens, skip_special_tokens=True)
    print(decoded_back)

    assert len(segments) == len(decoded_back)
    for i in range(len(segments)):
        assert segments[i]["transcription"] == decoded_back[i]["transcription"]
        if segments[i].get("spliced", None) != "left":
            assert tokenizer.timestamp_to_token(
                segments[i]["start"]
            ) == tokenizer.timestamp_to_token(decoded_back[i]["start"])
        if segments[i].get("spliced", None) != "right":
            assert tokenizer.timestamp_to_token(
                segments[i]["end"]
            ) == tokenizer.timestamp_to_token(decoded_back[i]["end"])
