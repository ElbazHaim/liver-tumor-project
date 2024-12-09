import uuid
from argparse import ArgumentParser
from dataclasses import dataclass

from data_preparation.transforms import train_transforms, val_transforms
from datasets import DatasetDict, load_from_disk
from transformers import (
    AutoConfig,
    AutoModelForImageClassification,
    AutoModelForSemanticSegmentation,
    HfArgumentParser,
    TrainingArguments,
)


def parse_args_from_yaml() -> tuple:
    """
    Parse the arguments from the yaml file and return the training arguments and other
    arguments.
    """
    parser = ArgumentParser()
    parser.add_argument("--yaml", type=str, required=True)
    ___args = parser.parse_args()

    @dataclass
    class OtherArgs:
        """
        Additional argumnents for convenience - these
        do not appear in the training argument parser.
        """

        # Filesystem args
        experiment_name: str
        checkpoint: str
        dataset: str

        # Early stopping args
        threshold: float
        patience: int

        # Utility args
        mlflow_uri: str
        id2label: dict
        device: str
        classification_model: bool

    hf_parser = HfArgumentParser((TrainingArguments, OtherArgs))
    training_args, args = hf_parser.parse_yaml_file(___args.yaml)

    return training_args, args


def add_uid(input_string: str) -> str:
    """
    Appends a 5-character unique identifier (UID) to the given string.

    The UID is derived from a UUID and truncated to the first 5 characters.

    Args:
        input_string (str): The input string to which the UID will be appended.

    Returns:
        str: The input string with the appended 5-character UID.
    """
    uid = str(uuid.uuid4())[:5]
    return input_string + "-" + uid


def load_model(
    checkpoint: str, id2label: dict, device: str
) -> AutoModelForSemanticSegmentation:
    """
    Load the model from the given directory and change its id2label mapping
    according to the given id2label argument.
    Args:
        checkpoint (str): The directory containing the model files.
        id2label_file (str): The path to the id2label file.
    Returns:
        model (AutoModelForSemanticSegmentation): The loaded model.
    """
    return AutoModelForSemanticSegmentation.from_pretrained(
        checkpoint,
        config=AutoConfig.from_pretrained(
            checkpoint, id2label=id2label, label2id={v: k for k, v in id2label.items()}
        ),
    ).to(device)


def load_training_and_validation_datasets(dataset: str) -> tuple[DatasetDict]:
    """
    Load the training and validation datasets from the given directory.
    Args:
        dataset (str): The directory containing the dataset files.
    Returns:
        Tuple[Dataset, Dataset]: The training and validation
        datasets loaded from the given directory.

    """
    dataset = load_from_disk(dataset)
    train_ds = dataset["train"]
    val_ds = dataset["val"]
    train_ds.set_transform(train_transforms)
    val_ds.set_transform(val_transforms)
    return train_ds, val_ds


def load_classification_model(
    checkpoint: str, id2label: dict, device: str
) -> AutoModelForImageClassification:
    """
    Load the model from the given directory and change its id2label mapping
    according to the given id2label argument.
    Args:
        checkpoint (str): The directory containing the model files.
        id2label_file (str): The path to the id2label file.
    Returns:
        model (AutoModelForImageClassification): The loaded model.
    """
    return AutoModelForImageClassification.from_pretrained(
        checkpoint,
        id2label=id2label,
        label2id={v: k for k, v in id2label.items()},
        num_labels=len(id2label.keys()),
    ).to(device)
