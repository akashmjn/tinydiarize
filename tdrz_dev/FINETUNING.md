
setup extra finetune_requirements
fetch data and runs from blob
    ./scripts/fetch_finetune_stuff.sh


notes about tricks:

memory reduction
    - fp16
    - activation checkpointing

different finetune modes (all, decoder, embeds)
freezing of token vocab
timestamp format (with/without, before/after)
