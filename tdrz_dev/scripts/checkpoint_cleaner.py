import glob
import os
import shutil
import sys
from pathlib import Path

print(os.getcwd())

# accept from command line
dirname = sys.argv[1]
# dirname = "whisper_finetuned-ami_train_nots_before/tiny.en"
# dirname = "whisper_finetuned_ami_train-tiny.en"

# recursively delete all checkpoint directories under dirname except for the one with the highest number
# e.g. ls ./tinydiarize/scratch/finetune_runs/all  -> checkpoint-200, checkpoint-400 .. checkpoint-1200
# we want to keep checkpoint-1200 and delete everything else
# and this should run recursively for all subdirectories under ./tinydiarize/scratch/finetune_runs/*

# get all subdirectories one level under dirname
subdirs = glob.glob(dirname + "/*")
print("\nFound the following subdirectories:")
print(subdirs)

# for each subdirectory, get all checkpoint directories
for subdir in subdirs:
    checkpoints = glob.glob(subdir + "/checkpoint-*")
    print("\nFound the following checkpoints in " + subdir + ":")
    print(checkpoints)

    # get the checkpoint with the highest number
    max_checkpoint = max(checkpoints, key=lambda x: int(x.split("-")[-1]))
    print("\Found the following checkpoint with max number:")
    print(max_checkpoint)

    # make a retry loop where we ask for confirmation before proceeding
    best_checkpoint = max_checkpoint
    while True:
        # ask for confirmation before proceeding
        print(
            "\nAre you sure you want to delete all checkpoints except "
            + best_checkpoint
            + "?"
        )
        print("Type y to proceed, or another checkpoint name to keep that one.")
        confirmation = input()
        if confirmation == "y":
            break
        elif confirmation.startswith("checkpoint-"):
            best_checkpoint = [x for x in checkpoints if x.endswith(confirmation)][0]
            continue
        else:
            print("Invalid input. Try again.")

    # delete all checkpoints except the one with the highest number
    for checkpoint in checkpoints:
        if checkpoint != best_checkpoint:
            shutil.rmtree(checkpoint)
