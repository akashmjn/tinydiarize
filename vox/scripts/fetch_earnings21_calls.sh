#!/bin/bash

# define an array of IDs of calls to download
IDS=("4385939" "4374910" "4359971" "4364366" "4341191")

# loop through the IDs
for i in ${IDS[@]};
do
    # URL of audio file
    URL=https://github.com/revdotcom/speech-datasets/blob/main/earnings21/media/$i.mp3?raw=true
    # use wget to download file from a URL
    wget $URL -O ../scratch/audio/earnings21-$i.mp3

    # URL of transcript file
    URL=https://github.com/revdotcom/speech-datasets/raw/main/earnings21/transcripts/nlp_references/$i.nlp
    # use wget to download file from a URL
    wget $URL -O ../fstalign_scoring/inputs/earnings21-$i-ref.nlp
done
