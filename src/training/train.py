"""
Module for training the model with arguments parsed from a yaml file.
The yaml file is accepted as a CLI argument and parsed using the HfArgumentParser.
Usage example:
$ python trainer.py --yaml path/to/args.yaml
"""

import evaluate
import mlflow
from training.utils import (
    add_uid,
    load_classification_model,
    load_model,
    parse_args_from_yaml,
    load_training_and_validation_datasets,
)
from transformers import (
    EarlyStoppingCallback,
    Trainer,
    TrainerCallback,
)
from transformers.integrations import MLflowCallback
from datasets import Dataset

metric = evaluate.load("mean_iou")


def train(
    model,
    training_args,
    callbacks: list[TrainerCallback] | None,
    train_dataset: Dataset,
    eval_dataset: Dataset,
) -> None:
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=callbacks,
    )
    with mlflow.start_run() as _:
        trainer.train()


def main() -> None:
    training_args, args = parse_args_from_yaml()
    train_ds, val_ds = load_training_and_validation_datasets(args.dataset)
    model = (
        load_classification_model(args.checkpoint, args.id2label, args.device)
        if args.classification_model
        else load_model(args.checkpoint, args.id2label, args.device)
    )
    mlflow.set_tracking_uri(args.mlflow_uri)
    training_args.output_dir = add_uid(args.experiment_name)
    train(
        model=model,
        training_args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        callbacks=[
            MLflowCallback,
            EarlyStoppingCallback(args.patience, args.threshold),
        ],
    )


if __name__ == "__main__":
    main()
