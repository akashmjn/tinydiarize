import argparse
import os
from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Union

import datasets
import torch
from tdrz_tokenizer import WhisperDiarizedTokenizer
from transformers import (
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    WhisperConfig,
    WhisperFeatureExtractor,
    WhisperForConditionalGeneration,
)

DATADIR = os.path.expanduser("~/.cache/huggingface/datasets")
WHISPERMODEL = "openai/whisper-small.en"

feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPERMODEL)
tokenizer = WhisperDiarizedTokenizer.from_pretrained(WHISPERMODEL)


# load dataset and pre-process for training
def prepare_dataset(batch):
    result = {}
    # flatten into a dict if called with .map(batch_size=1)
    if isinstance(batch["audio"], list):
        for k in batch.keys():
            assert len(batch[k]) == 1  # batch size is 1
            batch[k] = batch[k][0]

    input_features = feature_extractor(
        batch["audio"]["array"], sampling_rate=batch["audio"]["sampling_rate"]
    ).input_features[0]

    tokenizer.set_prefix_tokens(predict_timestamps=True)
    tokens_w_timestamps = tokenizer.encode(
        batch["segments"], add_speaker_turns="before"
    )

    tokenizer.set_prefix_tokens(predict_timestamps=False)
    tokens_wo_timestamps = tokenizer.encode(
        batch["segments"], add_speaker_turns="before"
    )

    # add both tokens_w_timestamps and tokens_wo_timestamps to batch
    result["input_features"] = [input_features, input_features.copy()]
    result["labels"] = [tokens_w_timestamps, tokens_wo_timestamps]

    return result


def prepare_model(model, finetune_mode="all", freeze_tokens="none"):
    model.config.use_cache = (
        False  # avoid repeated warning logs when using gradient checkpointing
    )

    if finetune_mode == "decoder":
        print("Freezing encoder params ..")
        model.freeze_encoder()
    elif "embed" in finetune_mode:
        p_names = [
            f"model.decoder.embed_{s}.weight" for s in finetune_mode.split("_")[1:]
        ]
        # freeze all params except for the token embedding
        print("Freezing all params except for embeddings ..", p_names)
        for name, param in model.named_parameters():
            if name in p_names:  # ["embed_tokens" / "embed_positions"]
                print(
                    "\t", name
                )  # implicitly also unfreezes model.proj_out which is tied
                continue
            param.requires_grad = False

    if freeze_tokens != "none":
        # register hook on param to zero gradients except for indices of freeze_tokens
        def hook_fn(grad, mode):
            if "bpe" in mode:
                grad[
                    : model.config.eos_token_id, :
                ] = 0.0  # TODO@Akash figure out why this is wrong in tokenizer
            if "time" in mode:
                grad[tokenizer.timestamp_begin :, :] = 0.0
            if mode == "spk_only":
                raise NotImplementedError
            return grad

        embed_tokens = model.get_decoder().embed_tokens.weight
        print(embed_tokens.shape)
        embed_tokens.register_hook(partial(hook_fn, mode=freeze_tokens))
        print("Added hook to zero gradients for embed_tokens for", freeze_tokens)
        # if "spk" in freeze_tokens:
        #     print(f"\tspeaker token: {tokenizer.speaker_turn_token_id}")
        # if "time" in freeze_tokens:
        #     print(f"\ttimestamp tokens: {tokenizer.timestamp_begin}-end")

    # enumerate over all model parameters and count % of trainable params
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Total params: {num_params}, Trainable params: {num_trainable_params}, % trainable: {num_trainable_params/num_params*100:.2f}%"
    )

    return model


# data collator
@dataclass
class DataCollatorSpeechSeq2SeqWithPadding:
    feature_extractor: Any
    tokenizer: Any

    def __call__(
        self, features: List[Dict[str, Union[List[int], torch.Tensor]]]
    ) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors
        input_features = [
            {"input_features": feature["input_features"]} for feature in features
        ]
        batch = self.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(
            labels_batch.attention_mask.ne(1), -100
        )

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch


class EvalFirstStepCallback(TrainerCallback):
    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step == 1 and args.logging_first_step:
            control.should_evaluate = True
        return control


if __name__ == "__main__":
    # parse command line arguments for training
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finetune_mode",
        type=str,
        default="all",
        choices=["all", "decoder", "embed_tokens", "embed_tokens_positions"],
    )
    # parser.add_argument("--finetune_tokens", type=str, default="all", choices=["all", "spk", "spk_time"])
    parser.add_argument(
        "--freeze_tokens",
        type=str,
        default="none",
        choices=["none", "bpe", "bpe_time", "spk_only"],
    )
    parser.add_argument(
        "--output_dir", type=str, default="whisper-tiny.en-finetuned/embed_tokens-1"
    )
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--max_steps", type=int, default=1000)
    parser.add_argument("--warmup_steps", type=int, default=200)
    parser.add_argument("--per_device_train_batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--gradient_checkpointing", type=bool, default=True)
    parser.add_argument("--fp16", type=bool, default=True)
    parser.add_argument("--evaluation_strategy", type=str, default="steps")
    parser.add_argument("--prediction_loss_only", type=bool, default=True)
    parser.add_argument("--eval_steps", type=int, default=200)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--logging_steps", type=int, default=50)
    parser.add_argument("--report_to", type=str, default="tensorboard")
    args = parser.parse_args()
    # convert args to dict
    args_dict = vars(args)
    args_dict["logging_first_step"] = True
    args_dict["per_device_eval_batch_size"] = args_dict["per_device_train_batch_size"]
    args_dict["dataloader_num_workers"] = 4

    # reference: https://huggingface.co/blog/fine-tune-whisper

    # load datasets and pre-process for training
    train_datasets = dict()
    for split in "train", "val":
        ds = datasets.load_from_disk(f"{DATADIR}/ami_{split}_chunked")
        # ds.cleanup_cache_files()  # NOTE: uncomment when you want reprocess/tokenize the dataset
        ds_for_train = ds.map(
            prepare_dataset,
            remove_columns=ds.column_names,
            num_proc=4,
            batched=True,
            batch_size=1,
        )
        train_datasets[split] = ds_for_train

    # data collator
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(feature_extractor, tokenizer)

    # loading model checkpoint (dont need to extend vocab as reusing startoflm token)
    model_config = WhisperConfig.from_pretrained(WHISPERMODEL)
    # model_config.dropout = 0.1              # TODO@Akash: fix in args
    # model_config.attention_dropout = 0.1
    model = WhisperForConditionalGeneration.from_pretrained(
        WHISPERMODEL, config=model_config
    )
    model = prepare_model(
        model, args_dict.pop("finetune_mode"), args_dict.pop("freeze_tokens")
    )

    # training args
    training_args = Seq2SeqTrainingArguments(**args_dict)

    # compute metrics for eval (skip for now)

    # trainer.train() :)
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_datasets["train"],
        eval_dataset=train_datasets["val"],
        data_collator=data_collator,
        tokenizer=feature_extractor,
        callbacks=[EvalFirstStepCallback()],
    )

    train_dataloader = trainer.get_train_dataloader()

    trainer.train()
