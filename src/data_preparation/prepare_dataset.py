"""
Module to process and prepare the liver tumor images dataset for training with
xshuggingface transformers library.
"""

import argparse
import logging
from pathlib import Path
from datasets import DatasetDict
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from multiprocessing import Process
from data_preparation.dataset_utils import (
    create_dataset_from_split_dir,
    process_and_save_volumes,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


def parse_args():
    """
    Parse command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Process and prepare medical imaging datasets."
    )
    parser.add_argument(
        "--data_root_dir",
        type=str,
        required=True,
        help="Path to the root data directory.",
    )
    parser.add_argument(
        "--arrow_output_dir",
        type=str,
        required=True,
        help="Path to save the Arrow format dataset.",
    )
    return parser.parse_args()


def load_file_paths(raw_volumes: Path, raw_segs: Path) -> list[dict]:
    """
    Load file paths for volumes and segmentations.
    """
    files_list = []
    num_volumes = len(list(raw_volumes.iterdir()))
    for vol_number in tqdm(range(num_volumes)):
        files_list.append(
            {
                "vol": raw_volumes / f"volume-{vol_number}.nii",
                "seg": raw_segs / f"segmentation-{vol_number}.nii",
            }
        )
    return files_list


def process_split(split_name: str, split: list, processed_data_path: Path) -> None:
    """
    Process and save a split of the dataset.
    Args:
        split_name (str): Name of the split.
        split (list): List of volumes and segmentations for the split.
        processed_data_path (Path): Path to save the processed dataset.
        Returns:
        None
    """
    split_path = processed_data_path / split_name
    logging.info(f"Processing: {split_path}")
    process_and_save_volumes(
        window_level=30,
        window_width=150,
        raw_paths=split,
        processed_volumes=split_path / "images",
        processed_seg=split_path / "labels",
    )


def create_datasets(data_root_dir: str) -> Path:
    """
    Create training, validation, and test datasets.
    Args:
        data_root_dir (str): Path to the root data directory.
    Returns:
        processed_data_path (Path): Path to the processed dataset.
    """
    raw_data_path = Path(data_root_dir) / "raw"
    processed_data_path = Path(data_root_dir) / "20241003_processed"

    raw_volumes = raw_data_path / "volumes"
    raw_segs = raw_data_path / "segmentations"

    files_list = load_file_paths(raw_volumes, raw_segs)

    train_val, test = train_test_split(files_list, test_size=0.2, random_state=420)
    train, val = train_test_split(train_val, test_size=0.1, random_state=420)

    logging.info(f"Training set size: {len(train)}")
    logging.info(f"Validation set size: {len(val)}")
    logging.info(f"Test set size: {len(test)}")

    logging.info("Creating datasets:")

    def run_processes():
        processes = [
            Process(target=process_split, args=("train", train, processed_data_path)),
            Process(target=process_split, args=("val", val, processed_data_path)),
            Process(target=process_split, args=("test", test, processed_data_path)),
        ]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

    run_processes()
    logging.info(
        "Finished processing dataset into PNG format. Creating Arrow format dataset."
    )
    return processed_data_path


def save_arrow_dataset(processed_data_path: Path, arrow_output_dir: Path) -> None:
    """
    Save the processed dataset to Arrow format.
    Args:
        processed_data_path (Path): Path to the processed dataset.
            arrow_output_dir (Path): Path to save the Arrow format dataset.
    """
    logging.info(
        f"Saving Arrow format dataset. Input data path: {processed_data_path}, output path: {arrow_output_dir}"
    )
    processed_splits = {
        "train": processed_data_path / "train",
        "val": processed_data_path / "val",
        "test": processed_data_path / "test",
    }

    dataset = DatasetDict(
        {
            split_name: create_dataset_from_split_dir(split_path)
            for split_name, split_path in processed_splits.items()
        }
    )

    dataset.save_to_disk(arrow_output_dir)


def main():
    args = parse_args()
    processed_data_path = create_datasets(args.data_root_dir)
    save_arrow_dataset(processed_data_path, args.arrow_output_dir)


if __name__ == "__main__":
    main()
