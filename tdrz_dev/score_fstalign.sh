# Description: Score a hypothesis file using fstalign to get WER, per-token metrics, and side-by-side analysis
# Usage: score_fstalign.sh <ref> <hyp> <outdir> [<wer-sidecar>]
#           (the --wer-sidecar option is not used in the current version of the code)
# Setup: docker pull revdotcom/fstalign  (more info at https://github.com/revdotcom/fstalign)


# these paths must fall within the current directory so that they can be mounted in the docker container
REF=$1
HYP=$2
OUTDIR=$3
FNAME=$(basename $HYP .nlp)

# set -x

# the current directory is mounted as /fstalign/workdir so all relative paths have to be relative to that
PREFIX="/fstalign/workdir"
CMD="/fstalign/build/fstalign wer \
    --ref $PREFIX/$REF --hyp $PREFIX/$HYP \
    --log $PREFIX/$OUTDIR/$FNAME.log --json-log $PREFIX/$OUTDIR/$FNAME.json --output-sbs $PREFIX/$OUTDIR/$FNAME.sbs"

# this argument doesn't really seem to be necessary
if ! [ -z "$4" ]
then
    CMD="$CMD --wer-sidecar $4"
fi

# command is run inside the docker container
docker run --rm -it -v $PWD:/fstalign/workdir revdotcom/fstalign $CMD

# print absolute path of output filename to stdout
echo "RESULT="$(realpath $OUTDIR/$FNAME.json)
