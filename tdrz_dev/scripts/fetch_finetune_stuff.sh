
cd ..

DATADIR=/mnt/scratch

wget https://sharedstorage7190.blob.core.windows.net/tinydiarize/precomputed/tdrz_ft_ami_prepared-hf_datasets.tar.gz -C $DATADIR
tar -zxvf $DATADIR/tdrz_ft_ami_prepared-hf_datasets.tar.gz -C $DATADIR
echo "Fetched finetuning dataset"

mkdir -p workdir_finetune
wget https://sharedstorage7190.blob.core.windows.net/tinydiarize/precomputed/workdir_finetune-03_09_23.tar.gz 
tar -zxvf workdir_finetune-03_09_23.tar.gz -C .
echo "Fetched finetune runs and checkpoints"

cd ./scripts
