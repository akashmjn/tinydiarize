import argparse
import logging
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

DATADIR = os.path.realpath("workdir_finetune/datasets")  # TODO@Akash - handle this in args
WHISPERMODEL = "openai/whisper-small.en"                   # TODO@Akash - handle this in args

feature_extractor = WhisperFeatureExtractor.from_pretrained(WHISPERMODEL)
tokenizer = WhisperDiarizedTokenizer.from_pretrained(WHISPERMODEL)
logger = logging.getLogger(__name__)


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


def prepare_model(model, finetune_mode="all", freeze_tokens="none"):
    model.config.use_cache = (
        False  # avoid repeated warning in logs when using gradient checkpointing
    )

    if finetune_mode == "decoder":  # option: decoder (only finetune decoder)
        logger.info("Freezing encoder params ..")
        model.freeze_encoder()
    elif "embed" in finetune_mode:  # options: embed_tokens, embed_tokens_positions
        # only finetune the token and/or position embeddings
        p_names = [
            f"model.decoder.embed_{s}.weight" for s in finetune_mode.split("_")[1:]
        ]
        logger.info("Freezing all params except for embeddings ..", p_names)
        for name, param in model.named_parameters():
            if name in p_names:  # ["embed_tokens" / "embed_positions"]
                logger.info(
                    "\t", name
                )  # implicitly also unfreezes model.proj_out which is tied
                continue
            param.requires_grad = False

    if freeze_tokens != "none":  # options: none, bpe, bpe_time, except_spk
        def hook_fn(grad, mode):
            # this gradient hook is used to zero gradients for a subset of tokens
            if "bpe" in mode:
                grad[
                    : model.config.eos_token_id, :
                ] = 0.0  # TODO@Akash figure why i've to use model.config and not tokenizer
            if "time" in mode:
                grad[tokenizer.timestamp_begin :, :] = 0.0
            if mode == "except_spk":
                grad[:tokenizer.speaker_turn_token_id, :] = 0.0
                grad[tokenizer.speaker_turn_token_id + 1 :, :] = 0.0
            return grad

        # apply gradient hook to token embeddings (and implicitiy tied output layer)
        embed_tokens = model.get_decoder().embed_tokens.weight
        logger.info(embed_tokens.shape)
        embed_tokens.register_hook(partial(hook_fn, mode=freeze_tokens))
        logger.info("Added hook to zero gradients for embed_tokens for", freeze_tokens)

    # enumerate over all model parameters and count % of trainable params
    num_params = sum(p.numel() for p in model.parameters())
    num_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        f"Total params: {num_params}, Trainable params: {num_trainable_params}, % trainable: {num_trainable_params/num_params*100:.2f}%"
    )

    return model


if __name__ == "__main__":
    # parse command line arguments for training
    # TODO@Akash take args from a yaml file
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--finetune_mode",
        type=str,
        default="decoder",
        choices=["all", "decoder", "embed_tokens", "embed_tokens_positions"],
    )
    parser.add_argument(
        "--freeze_tokens",
        type=str,
        default="bpe_time",
        choices=["none", "bpe", "bpe_time", "except_spk"],
    )
    parser.add_argument(
        "--output_dir", type=str, default="workdir_finetune/finetune_runs"
    )
    parser.add_argument("--learning_rate", type=float, default=1.75e-5)
    parser.add_argument("--max_steps", type=int, default=3000)
    parser.add_argument("--warmup_steps", type=int, default=400)
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
    # convert args to dict  # TODO@Akash: save these args in a config in the output dir
    args_dict = vars(args)
    args_dict["output_dir"] = f'{args_dict["output_dir"]}/{WHISPERMODEL}/{args_dict["finetune_mode"]}-{args_dict["freeze_tokens"]}'
    args_dict["logging_first_step"] = True
    args_dict["per_device_eval_batch_size"] = args_dict["per_device_train_batch_size"]
    args_dict["dataloader_num_workers"] = 4
    logger.setLevel(logging.INFO)

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
    model_config = WhisperConfig.from_pretrained(WHISPERMODEL, apply_spec_augment=True)
    # model_config.dropout = 0.1
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

    # finally run trainer.train() :)
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
