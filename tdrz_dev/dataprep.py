import argparse
import os
import string
from collections import defaultdict
from dataclasses import dataclass

from datasets import load_dataset
from pyannote.core import Annotation, Segment


# function to query list of words from example_words using (speaker, start_time, end_time)
def query_words(start_time, end_time, words_list):
    words = []
    for w in words_list:
        if w["word"] in string.punctuation:
            if len(words) == 0:  # ignore punctuations at the start of a segment
                continue
            elif (
                words[-1]["word"] not in string.punctuation
            ):  # always include punc after a word
                words.append(w)
        elif w["start"] >= start_time and w["end"] <= end_time:  # include full words
            words.append(w)
        # elif w["start"] < end_time < w["end"]:  # include partially right-truncated words
        #     w = w.copy()
        #     w.update({"end": end_time})
        #     words.append(w)
        elif w["end"] > end_time:
            break
    return words


def join_words(words):
    if len(words) > 0:
        return "".join(
            [
                w["word"] if w["word"] in string.punctuation else " " + w["word"]
                for w in words
            ]
        ).strip()
    else:
        return "EMPTY"


@dataclass(frozen=True, order=True)
class AnnotatedSegment(Segment):
    start: float = 0.0
    end: float = 0.0
    speaker: str = ""
    transcription: str = ""
    words: list = None

    def __str__(self):
        """Human-readable representation

        >>> print(Segment(1337, 1337 + 0.42))
        [ 00:22:17.000 -->  00:22:17.420]

        Note
        ----
        Empty segments are printed as "[]"
        """
        if self:
            return "[%s --> %s]\t%s\t%s" % (
                self._str_helper(self.start),
                self._str_helper(self.end),
                self.speaker,
                self.transcription,
            )

        return "[]"

    def __repr__(self):
        return f"AnnotatedSegment({self.start}, {self.end}, {self.speaker}, {self.transcription}, words[{self.n_words()}])"

    def __and__(self, other):
        if not isinstance(other, Segment):
            return NotImplemented
        start = max(self.start, other.start)
        end = min(self.end, other.end)
        return self.splice(start, end)

    def n_words(self):
        return len(self.words) if self.words is not None else 0

    def for_json(self, offset=0.0):
        return dict(
            start=self.start + offset,
            end=self.end + offset,
            speaker=self.speaker,
            transcription=self.transcription,
            words=self.words,
        )

    @classmethod
    def from_json(cls, data):
        return cls(
            data["start"],
            data["end"],
            data["speaker"],
            data["transcription"],
            data["words"],
        )

    # splice by time to create subsegments
    def splice(self, start, end):
        assert start >= self.start and end <= self.end
        words = query_words(start, end, self.words)
        if len(words) == 0:
            return AnnotatedSegment(start, end, self.speaker, None)
        else:
            spliced_start = words[0]["start"]
            spliced_end = words[-1]["end"]
            return AnnotatedSegment(
                spliced_start, spliced_end, self.speaker, join_words(words), words
            )


"""
Populate pyannote data structures with segment information in ami_example:

ami_example schema:
{
    'word_ids', 'word_start_times', 'word_end_times', 'word_speakers', 'words',
    'segment_ids', 'segment_start_times', 'segment_end_times', 'segment_speakers', 'channels', 'file', 'audio'
}
"""


def example_to_annotation(data_example):
    # create pyannote Annotation object (internally uses pyannote.core.Segment)
    # Annotation internally sorts the segments by start time
    annotation = Annotation()
    for i in range(len(data_example["segment_ids"])):
        start, end = (
            data_example["segment_start_times"][i],
            data_example["segment_end_times"][i],
        )
        annotation[Segment(start, end)] = data_example["segment_speakers"][i]

    # TODO@Akash - print some statistics e.g. number of speakers, number of segments, overlap percentage etc.
    return annotation


def split_segments(data_example, annotation, verbose=False):
    # Iterate over words in example and save into a dict keyed by speaker
    example_words = defaultdict(list)
    for i in range(len(data_example["word_ids"])):
        start, end = (
            data_example["word_start_times"][i],
            data_example["word_end_times"][i],
        )
        example_words[data_example["word_speakers"][i]].append(
            dict(word=data_example["words"][i], start=start, end=end)
        )

    # print out the segments in order
    segments = []
    start, end = 0.0, 0.0
    for i, (a_segment, _, spk) in enumerate(annotation.itertracks(yield_label=True)):
        assert round(a_segment.start, 2) >= round(
            start, 2
        ), f"non monotonic segment {i}: {a_segment}\nstart {start:.2f} -> {a_segment.start:.2f}"  # segments are sorted by start time
        if verbose:
            if a_segment.start < end:
                print("overlap")
        start = round(a_segment.start, 2)
        end = round(a_segment.end, 2)
        words = query_words(start, end, example_words[spk])

        if len(words) == 0:
            segment_str = f"EMPTY__{start:.2f}__{end:.2f}"
        else:
            segment_str = join_words(words)

        segment = AnnotatedSegment(start, end, spk, segment_str, words)
        if len(segment.words) > 0:
            segments.append(segment)

        if verbose:
            print(segment)

    return segments


def split_chunks(segments, chunk_size=30.0, drop_words=False):
    """
    Split segments into 30s chunks
    chunk transcription will contain <st> between segments from different speakers
    partially contained segments are spliced to fit into the chunk

    # NOTE TODO@Akash: if there are several overlapped partially contained segments,
    # only first gets added, rest will get skipped

    schema for each chunk:
        [start, end, transcription, segments:[{start, end, speaker, transcription, words}, ...]]
    """

    def _start_new_chunk(start):
        return dict(start=start, end=start + chunk_size, segments=[]), Segment(
            start, start + chunk_size
        )

    def _add_spliced_segment(chunk, chunk_window, segment, side="right"):
        spliced_segment = segment & chunk_window
        if spliced_segment.n_words() > 0:
            chunk["segments"].append(
                spliced_segment.for_json(offset=-chunk_window.start)
            )
            chunk["segments"][-1]["spliced"] = side
        if side == "right":
            return spliced_segment.end
        else:
            return spliced_segment.start

    chunks = []
    current_chunk, chunk_window = _start_new_chunk(0.0)

    for i, segment in enumerate(segments):
        if segment in chunk_window:  # completely contained in chunk
            current_chunk["segments"].append(
                segment.for_json(offset=-chunk_window.start)
            )

        elif segment.overlaps(chunk_window.end):  # extends into next chunk
            # add the overlapping part of segment to current chunk
            splice_point = _add_spliced_segment(
                current_chunk, chunk_window, segment, side="right"
            )

            # end this chunk start a new one
            chunks.append(current_chunk)
            current_chunk, chunk_window = _start_new_chunk(splice_point)

            # add the remaining part of segment to new chunk
            _add_spliced_segment(current_chunk, chunk_window, segment, side="left")

        elif segment.overlaps(chunk_window.start):  # extends into previous chunk
            # add to the previous chunk
            prev_chunk, prev_chunk_window = chunks[-1], Segment(
                chunks[-1]["start"], chunks[-1]["end"]
            )
            _add_spliced_segment(prev_chunk, prev_chunk_window, segment, side="left")
            _add_spliced_segment(current_chunk, chunk_window, segment, side="right")

            print(f"NOTE: add segment {i} to previous chunk")
            print(f"\t{'Windows:':<10s}", prev_chunk_window, chunk_window)
            print(f"\t{'Segment:':<10s}", segment)

        elif segment.start >= chunk_window.end:
            # end this chunk start a new one
            chunks.append(current_chunk)
            current_chunk, chunk_window = _start_new_chunk(
                chunk_window.start + chunk_size
            )

        else:
            print(f"WARNING: segment {i} was skipped...")
            # print window and segment in tabular format
            print(f"\t{'Window:':<10s}", chunk_window)
            print(f"\t{'Segment:':<10s}", segment)

    # last one
    if len(current_chunk["segments"]) > 0:
        chunks.append(current_chunk)

    if drop_words:
        for c in chunks:
            for s in c["segments"]:
                s.pop("words")

    # assert that start, end times for segments are bounded within [0., 30.] seconds
    for i, c in enumerate(chunks):
        for j, s in enumerate(c["segments"]):
            try:
                assert (
                    0.0 <= s["start"] <= chunk_size
                ), f"segment start time is out of bounds: {s['start']} in chunk {i} segment {j}"
                assert (
                    0.0 <= s["end"] <= chunk_size
                ), f"segment end time is out of bounds: {s['end']} in chunk {i} segment {j}"
            except AssertionError as e:
                print(e)
                import pdb

                pdb.set_trace()

    return chunks


# combine transcriptions with <st> between segments from different speakers
def make_chunk_transcription(chunk):
    ST_TOKEN = "<|st|>"
    time_token = lambda x: f"<|{round(x / 0.02) * 0.02:.2f}|>"
    if len(chunk["segments"]) == 0:
        return ""
    transcription = []
    speaker = chunk["segments"][0]["speaker"]
    for s in chunk["segments"]:
        if s["speaker"] != speaker:
            transcription.append(ST_TOKEN)
        # if s.get("spliced", None)!="left":
        transcription.append(time_token(s["start"]))  # always add start time
        transcription.append(s["transcription"])
        if (
            s.get("spliced", None) != "right"
        ):  # end time prediction used for long-form inference
            transcription.append(time_token(s["end"]))
    return " ".join(transcription).strip()


def print_chunk(chunk):
    print(f'chunk {Segment(chunk["start"], chunk["end"])}')
    print(make_chunk_transcription(chunk) + "\n")
    for s in chunk["segments"]:
        print(f"\t{AnnotatedSegment.from_json(s)}")


# huggingface datasets map function ami_example -> chunks (transcript including <st> token + timestamps)
def chunk_example(data_example):
    # when running inside dataset.map, the example format is nested
    if len(data_example["audio"]) == 1:
        # flatten fields in data_example: list(len(1)) -> value
        for k in data_example:
            assert len(data_example[k]) == 1  # when batch_size=1
            data_example[k] = data_example[k][0]
    annotation = example_to_annotation(data_example)
    segments = split_segments(data_example, annotation)
    chunks = split_chunks(segments, drop_words=True)

    meeting_duration = (
        data_example["segment_end_times"][-1] - data_example["segment_start_times"][0]
    )
    chunks_duration = len(chunks) * 30.0
    assert (
        abs(meeting_duration - chunks_duration) / meeting_duration < 0.2
    ), f"Discrepancy: meeting_duration={meeting_duration:.1f} chunks_duration={chunks_duration:.1f}"

    for c in chunks:
        # add meeting_id
        c["meeting_id"] = data_example["word_ids"][0].split(".")[0]
        # add transcription
        c["transcription"] = make_chunk_transcription(c)
        # add audio
        sr = data_example["audio"]["sampling_rate"]
        c["audio"] = {}
        c["audio"]["array"] = data_example["audio"]["array"][
            round(c["start"] * sr) : round(c["end"] * sr)
        ].copy()
        c["audio"]["sampling_rate"] = sr
    # switch to columnar i.e. keys have list of values
    chunks = {k: [d[k] for d in chunks] for k in chunks[0].keys()}
    return chunks


if __name__ == "__main__":
    # accept command line arguments (split, datadir)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--split", type=str, default=None, required=True, help="split to process"
    )
    parser.add_argument(
        "--datadir",
        type=str,
        default="~/.cache/huggingface/datasets",
        help="directory to save the chunked dataset",
    )
    args = parser.parse_args()

    # load AMI dataset from huggingface datasets
    print("Loading AMI dataset...")
    ami_dataset = load_dataset("ami", "microphone-single")
    print(ami_dataset)

    split = args.split
    datadir = os.path.expanduser(f"{args.datadir}/ami_{split}_chunked")

    # process and save the chunked dataset
    print(f"Split: {split}")
    print(f"Datadir: {datadir}")
    if input("Create and save chunked dataset? (y/n)") != "y":
        exit()
    ami_dataset_split = ami_dataset[split]
    ami_dataset_split_chunked = ami_dataset_split.map(
        chunk_example,
        batched=True,
        batch_size=1,
        remove_columns=ami_dataset_split.column_names,
    )
    os.makedirs(datadir, exist_ok=True)
    ami_dataset_split_chunked.save_to_disk(datadir)
