
setup extra finetune_requirements
fetch data and runs from blob
    ./scripts/fetch_finetune_stuff.sh

notes about tricks:
    - memory reduction:
        (fp16, activation checkpointing)
    - regularization:
        (decoder only, vocab freezing, specaug)
    - dataprep: 
        with/without timestamp
        (other edge cases: better handling overlapped cross chunk boundary, overlap serialization, 
         dropping very long chunks, timestamp format too long)
        tbd: previous context
