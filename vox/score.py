import json
import re
import sys
import subprocess
import argparse
import os
from glob import glob
from pathlib import Path
from copy import deepcopy

DESCRIPTION = """
Score whisper reco with speaker turns added.
WER and speaker turn errors are computed using revdotcom/fstalign via edit distance of ref/reco transcripts.
For speaker turns errors, transcripts are re-aligned with a special token
inserted and we use token-level precision/recall errors exposed by the tool.
"""


# need to replace <|speakerturn|> since fstalign treats '|' as a separator
SCRIPT = "score_fstalign.sh"
WHISPER_ST_TOKEN = "<|speakerturn|>"
ST_TOKEN = "SPEAKER__TURN"
PUNCTUATION = set(['.', '?', '!', ',', ';'])
ENDING_PUNCTUATION = set(['.', '?', '!'])


# TODO@Akash - replace with proper whisper normalizer/tokenizer
# returns list of tuples (token, trailing punctuation)
def _tokenize_line(line):
    # handle "word.<|speakerturn|>", "word. <|speakerturn|> word", "word. <|speakerturn|>word"
    line = line.replace(WHISPER_ST_TOKEN, " "+ST_TOKEN+" ")
    # split into tokens
    tokens = [ t for t in re.split(r'[-\s]', line) if len(t.strip())>0 ]

    # handle ending punctuation
    result = []
    for t in tokens:
        if t[-1] in PUNCTUATION:
            if len(t)==1:
                print(f"WARNING: skipping token with only punctuation: `{t}` in line: `{line}`")
                continue
            p = t[-1]
            t = t[:-1]
        else:
            p = ''
        result.append((t, p))
    
    return result


# take earnings nlp file, convert and add speaker turn tags, remove extra entity tags
def tag_ref_nlp(nlp_file, output_dir=None, tag_speaker_turn=True):
    with open(nlp_file) as fp:
        raw_lines = fp.read().splitlines()

    suffix = '_tagged.nlp' if tag_speaker_turn else '.nlp'
    fname = Path(nlp_file).name.replace('.nlp', suffix)
    output_dir = Path(nlp_file).parent if output_dir is None else Path(output_dir)
    output_file = output_dir / fname
    n = 0
    with open(output_file, "w") as fp:
        for i, line in enumerate(raw_lines):
            if i==0:
                fp.write(line+'\n')
                continue
            elif i==1:
                speaker = line.split('|')[1]

            if line.split('|')[1] != speaker and tag_speaker_turn:
                # add the speaker turn tag
                fp.write(f"{ST_TOKEN}|0||||LC|['{n}:SPEAKER_TURN']|['{n}']"+'\n')  # entity ids must be unique
                n += 1

            # replace all with blank tags
            line = "|".join(line.split('|')[:6]) + "|[]|[]"
            fp.write(line+'\n')
            speaker = line.split('|')[1]

    print(f"Written {i+n} lines to {output_file}")            


# take whisper reco json, write out in nlp format, with speaker turn tokens added in two modes -
# one where speaker turn are added at segment boundaries, and one where they are added after every [.?!] punctuation token
def whisper_reco_to_nlp(reco_json, output_dir=None, speaker_turn_mode='segment'):
    """
    token|speaker|ts|endTs|punctuation|case|tags
    good|1|||||
    """
    with open(reco_json) as fp:
        reco = json.load(fp)

    suffix = f'_spkturn_{speaker_turn_mode}.nlp' if speaker_turn_mode else '.nlp'
    # suffix = '_spktagged.nlp' if tag_speaker_turn else '.nlp'
    fname = Path(reco_json).name.replace('.mp3', '').replace('.wav', '').replace('.json', suffix)
    output_dir = Path(reco_json).parent if output_dir is None else Path(output_dir)
    output_file = output_dir / fname

    n = 0
    with open(output_file, "w") as fp:
        fp.write("token|speaker|ts|endTs|punctuation|case|tags\n"); n+=1
        for i, segment in enumerate(reco["segments"]):
            if i == 0 and speaker_turn_mode == 'segment':
                curr_speaker = segment.get('speaker', '')

            if speaker_turn_mode == 'segment':
                # if speakers are provided, add speaker turn if speaker differs from previous one
                if 'speaker' in segment:
                    if segment['speaker'] != curr_speaker:
                        fp.write(f"{ST_TOKEN}|0|||||\n"); n+=1
                    curr_speaker = segment['speaker']
                # else default add speaker turn before every new segment
                elif n>0:
                    fp.write(f"{ST_TOKEN}|0|||||\n"); n+=1

            for token, punc in _tokenize_line(segment["text"]):
                if token == ST_TOKEN:
                    if speaker_turn_mode and 'token' in speaker_turn_mode:
                        fp.write(f"{ST_TOKEN}|0|||{punc}||\n"); n+=1
                    # strips out speaker turn tokens if speaker_turn_mode is None
                    continue
                fp.write(f"{token}|0|||{punc}||\n"); n+=1
                # add speaker turn after every punctuation token
                if speaker_turn_mode and 'punctuation' in speaker_turn_mode:
                    if punc in ENDING_PUNCTUATION:
                        fp.write(f"{ST_TOKEN}|0|||||\n"); n+=1
            
    print(f"Written {n} lines to {output_file}")
    return output_file


# function to read an .nlp file and strip out all speaker turn tokens
def strip_speaker_turn_tokens(nlp_file, output_dir=None):
    with open(nlp_file) as fp:
        raw_lines = fp.read().splitlines()

    fname = Path(nlp_file).name.replace('.nlp', '_for_wer.nlp')
    output_dir = Path(nlp_file).parent if output_dir is None else Path(output_dir)
    output_file = output_dir / fname
    n = 0
    with open(output_file, "w") as fp:
        for i, line in enumerate(raw_lines):
            if i==0:
                fp.write(line+'\n'); n += 1
                continue
            
            if line.split('|')[0] == ST_TOKEN:
                continue

            fp.write(line+'\n'); n += 1

    print(f"Written {i+n} lines to {output_file}")            

    return output_file


def process_result(wer_json, speaker_turn_json):
    with open(wer_json) as fp:
        wer_result = json.load(fp)
    with open(speaker_turn_json) as fp:
        speaker_turn_result = json.load(fp)
    
    processed_result = dict(wer_overall={}, speaker_turn={})

    # WER results
    processed_result["wer_overall"] = deepcopy(wer_result['wer']['bestWER'])
    for k in ['meta', 'precision', 'recall']:
        processed_result["wer_overall"].pop(k)
    processed_result["wer_speaker_switch"] = deepcopy(wer_result['wer']['speakerSwitchWER'])
    processed_result["wer_speaker_switch"].pop('meta')

    # speaker turn results
    speaker_turn = deepcopy(speaker_turn_result['wer']['unigrams'][ST_TOKEN.lower()])
    
    # account for errors esp substitution_fp
    # TODO@Akash - double check this logic, how to account for errors esp substitution_fp
    speaker_turn['substitutions'] = sum([speaker_turn[k] for k in ['substitutions_fn', 'substitutions_fp']])
    speaker_turn['numWordsInReference'] = sum([speaker_turn[k] for k in ['correct', 'deletions', 'substitutions_fn']])
    speaker_turn['numErrors'] = sum([speaker_turn[k] for k in ['substitutions', 'insertions', 'deletions']])
    # for details in logging 
    speaker_turn['numPredictions'] = sum([speaker_turn[k] for k in ['correct', 'insertions', 'substitutions_fp']])

    processed_result['speaker_turn'] = speaker_turn

    result_json = str(Path(speaker_turn_json).parent / ("result-" + Path(speaker_turn_json).name))
    with open(result_json, 'w') as fp:
        json.dump(processed_result, fp, indent=4)

    return result_json, processed_result


def summarize_result(result_json):
    with open(result_json) as fp:
        result = json.load(fp)

    wer = result['wer_overall']
    wer_spk = result['wer_speaker_switch']
    speaker_turn = result['speaker_turn']

    # TODO@Akash - make this 2 pandas dataframes

    print("\n\n"+"-"*50)
    print(f"Results for: {result_json}")
    print(f"Side-by-side analysis file at: {result_json.replace('result-', '').replace('.json', '.sbs')}\n")

    print(f'WER: {wer["wer"]:.4f} ({wer["numErrors"]}/{wer["numWordsInReference"]})')
    print(f'\tErrors: {wer["substitutions"]}/{wer["insertions"]}/{wer["deletions"]} [S/I/D]')
    print(f'Speaker switch WER: {wer_spk["wer"]:.4f} ({wer_spk["numErrors"]}/{wer_spk["numWordsInReference"]})')
    print(f'\tErrors: {wer_spk["substitutions"]}/{wer_spk["insertions"]}/{wer_spk["deletions"]} [S/I/D]')

    print(f'Speaker turn Precision: {speaker_turn["precision"]:.2f} ({speaker_turn["correct"]}/{speaker_turn["numPredictions"]})')
    print(f'\tErrors: {speaker_turn["substitutions_fp"]}/{speaker_turn["insertions"]} [S/I]')
    print(f'Speaker turn Recall: {speaker_turn["recall"]:.2f} ({speaker_turn["correct"]}/{speaker_turn["numWordsInReference"]})')
    print(f'\tErrors: {speaker_turn["substitutions_fn"]}/{speaker_turn["deletions"]} [S/D]')

    print("-"*50)


def score_fstalign(ref_nlp, reco_file, work_dir="./fstalign_scoring", speaker_turn_mode="segment"):

    inputs_dir = Path(work_dir)/"inputs"
    os.makedirs(inputs_dir, exist_ok=True)
    ref_nlp_for_wer = strip_speaker_turn_tokens(ref_nlp, inputs_dir)

    if Path(reco_file).suffix == '.nlp':
        reco_nlp = reco_file
        reco_nlp_for_wer = strip_speaker_turn_tokens(reco_nlp, inputs_dir)
    elif Path(reco_file).suffix == '.json':
        # convert to nlp format
        print("Converting reco to nlp format for scoring")
        reco_nlp = whisper_reco_to_nlp(reco_file, inputs_dir, speaker_turn_mode=speaker_turn_mode)
        reco_nlp_for_wer = whisper_reco_to_nlp(reco_file, inputs_dir, speaker_turn_mode=None)

    # we need to call fstalign twice, once for WER and once for speaker turn errors
    output_dir = str(Path(work_dir)/"results")

    # TODO@Akash - make this neat
    result = subprocess.check_output(['sh', SCRIPT, ref_nlp_for_wer, reco_nlp_for_wer, output_dir]).decode('utf-8').splitlines()[-1]
    assert result.startswith('RESULT='), "Unexpected output from "+SCRIPT
    wer_json = result.split('=')[1]
    assert os.path.exists(wer_json), "WER result file not found"
    # run scoring script on files with speaker turn tokens for precision/recall
    result = subprocess.check_output(['sh', SCRIPT, ref_nlp, reco_nlp, output_dir]).decode('utf-8').splitlines()[-1]
    assert result.startswith('RESULT='), "Unexpected output from "+SCRIPT
    speaker_turn_json = result.split('=')[1]
    assert os.path.exists(speaker_turn_json), "speaker_turn result file not found"

    # process result
    return process_result(wer_json, speaker_turn_json)


if __name__ == "__main__":

    print("NOTE: dont forget to pass the glob pattern inside quotes")
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    parser.add_argument("glob_pattern", help="glob pattern for reco files")
    parser.add_argument("--ref_nlp", help="reference nlp file", default="./fstalign_scoring/inputs/earnings21-4341191-ref_tagged.nlp")
    parser.add_argument("--work_dir", help="working directory for fstalign scoring", default="./fstalign_scoring")
    parser.add_argument("--speaker_turn_mode", help="speaker turn mode", choices=["segment", "punctuation", "token", "punctuation_token"], default="punctuation")
    args = parser.parse_args()

    ref_nlp = args.ref_nlp
    glob_pattern = args.glob_pattern

    reco_files = glob(glob_pattern)
    if input(f"Scoring {len(reco_files)} files under: {glob_pattern}, [y/n]\t").lower()=='n':
        exit()

    result_jsons = []
    for reco_file in reco_files:
        result_json, _ = score_fstalign(ref_nlp, reco_file, work_dir=args.work_dir, speaker_turn_mode=args.speaker_turn_mode)
        result_jsons.append(result_json)
    
    for result_json in result_jsons:
        summarize_result(result_json)
