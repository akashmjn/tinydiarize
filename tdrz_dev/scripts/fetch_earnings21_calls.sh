#!/bin/bash

# raise an error if WORKDIR is not set
if [ -z "$1" ]; then
    echo "WORKDIR is not set"
    exit 1
fi

AUDIODIR=$1/audio
REFDIR=$1/fstalign_scoring/references
mkdir -p $AUDIODIR $REFDIR

# define an array of IDs of calls to download
IDS=("4385939" "4374910" "4359971")

# loop through the IDs
for i in ${IDS[@]};
do
    # URL of audio file
    URL=https://github.com/revdotcom/speech-datasets/blob/main/earnings21/media/$i.mp3?raw=true
    # use wget to download file from a URL
    wget $URL -O $AUDIODIR/earnings21-$i.mp3

    # URL of transcript file
    URL=https://github.com/revdotcom/speech-datasets/raw/main/earnings21/transcripts/nlp_references/$i.nlp
    # use wget to download file from a URL
    wget $URL -O $REFDIR/earnings21-$i-ref.nlp
done
