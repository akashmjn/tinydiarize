#!/bin/sh

# setup for azcopy https://aka.ms/downloadazcopy-v10-linux, much faster download than wget
if ! command -v azcopy &> /dev/null
then
    echo "azcopy could not be found, installing"
    wget https://aka.ms/downloadazcopy-v10-linux -O azcopy.tar.gz
    tar -zxvf azcopy.tar.gz -C $PWD
    AZCOPY_DIR=$(realpath $(ls -d $PWD/azcopy_linux*))
    export PATH=$PATH:$AZCOPY_DIR
    echo "Fetched executable at $AZCOPY_DIR"
fi

cd ..

WORKDIR=$PWD/workdir_finetune

mkdir -p $WORKDIR

azcopy cp https://sharedstorage7190.blob.core.windows.net/tinydiarize/precomputed/tdrz_ft_ami_prepared-hf_datasets.tar.gz $WORKDIR/
tar -zxvf $WORKDIR/tdrz_ft_ami_prepared-hf_datasets.tar.gz -C $WORKDIR
echo "Fetched finetuning dataset"

azcopy cp https://sharedstorage7190.blob.core.windows.net/tinydiarize/precomputed/workdir_finetune-03_09_23.tar.gz $WORKDIR/
tar -zxvf $WORKDIR/workdir_finetune-03_09_23.tar.gz -C $WORKDIR
echo "Fetched finetune runs and checkpoints"

cd ./scripts
