"""
Module for processing liver segmentaion images in NII format.
"""

import logging
from pathlib import Path
import nibabel as nib
import numpy as np
import pandas as pd


def pad_or_trim(
    image: np.ndarray, segment: np.ndarray, target_depth: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Adjusts the depth of the input image and segment to match the target depth by either padding or trimming.

    Args:
        image (np.ndarray): The input image array.
        segment (np.ndarray): The input segmentation array.
        target_depth (int): The target depth to which the image and segment should be adjusted.

    Returns:
        Tuple[np.ndarray, np.ndarray]: The adjusted image and segmentation arrays.
    """
    current_depth = image.shape[2]

    if current_depth > target_depth:
        start_idx = (current_depth - target_depth) // 2
        image = image[:, :, start_idx : start_idx + target_depth]
        segment = segment[:, :, start_idx : start_idx + target_depth]
    elif current_depth < target_depth:
        pad_before = (target_depth - current_depth) // 2
        pad_after = target_depth - current_depth - pad_before
        image = np.pad(image, ((0, 0), (0, 0), (pad_before, pad_after)), "constant")
        segment = np.pad(segment, ((0, 0), (0, 0), (pad_before, pad_after)), "constant")

    return image, segment


def window_image(
    image: np.ndarray, window_level: float, window_width: float
) -> np.ndarray:
    """
    Apply windowing to a CT image for better visualization.

    Args:
        image (np.ndarray): The input CT image as a NumPy array.
        window_level (float): The center of the windowing range.
        window_width (float): The width of the windowing range.

    Returns:
        np.ndarray: The windowed image normalized to the range [0, 255] as a NumPy array.
    """
    min_intensity: float = window_level - (window_width / 2)
    max_intensity: float = window_level + (window_width / 2)

    windowed_image = np.clip(image, min_intensity, max_intensity)
    windowed_image = (
        (windowed_image - min_intensity) / (max_intensity - min_intensity) * 255
    ).astype(np.uint8)

    return windowed_image


def process_image_and_segment(
    img_path: Path,
    seg_path: Path,
    window_level: float,
    window_width: float,
    target_depth: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes the image and segmentation mask by adjusting their depth and applying windowing.

    Args:
        img_path (Path): Path to the image file.
        seg_path (Path): Path to the segmentation mask file.
        target_depth (int): Target depth for the images.
            Defaults to not enforcing a fixed size.
        window_level (float): Window level for image windowing.
        window_width (float): Window width for image windowing.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: The processed image, processed
        segmentation, and the affine matrix.
    """
    img_nii = nib.load(img_path)
    seg_nii = nib.load(seg_path)

    image = img_nii.get_fdata()
    segment = seg_nii.get_fdata()
    segment = np.uint8(segment)
    affine = img_nii.affine

    windowed_image = window_image(image, window_level, window_width)

    if target_depth:
        processed_image, processed_segment = pad_or_trim(
            windowed_image, segment, target_depth
        )
        return processed_image.T, processed_segment.T, affine.T

    return windowed_image, segment, affine


def save_processed_files(
    processed_image: np.ndarray,
    processed_segment: np.ndarray,
    affine: np.ndarray,
    img_path: Path,
    seg_path: Path,
    output_image_dir: Path,
    output_segment_dir: Path,
) -> None:
    """
    Saves the processed image and segmentation mask to the specified directories.

    Args:
        processed_image (np.ndarray): The processed image array.
        processed_segment (np.ndarray): The processed segmentation array.
        affine (np.ndarray): The affine transformation matrix.
        img_path (Path): Original image file path for naming the output file.
        seg_path (Path): Original segmentation file path for naming the output file.
        output_image_dir (Path): Directory to save processed volume files.
        output_segment_dir (Path): Directory to save processed segment files.
    """
    img_filename = img_path.name
    seg_filename = seg_path.name

    processed_img_path = output_image_dir / img_filename
    processed_seg_path = output_segment_dir / seg_filename

    nib.save(nib.Nifti1Image(processed_image, affine), processed_img_path)
    nib.save(nib.Nifti1Image(processed_segment, affine), processed_seg_path)

    logging.info(f"Processed and saved {img_filename} and {seg_filename}")


def process_and_save_files(
    csv_file: str | Path,
    output_image_dir: str | Path,
    output_segment_dir: str | Path,
    target_depth: int,
    window_level: float,
    window_width: float,
) -> None:
    """
    Processes and saves 3D medical images and segmentation masks, adjusting their depth
    and applying windowing.

    Args:
        csv_file (Union[str, Path]): Path to the CSV file containing volume and segment
        paths.
        output_image_dir (Union[str, Path]): Directory to save processed volume files.
        output_segment_dir (Union[str, Path]): Directory to save processed segment
                files.
        target_depth (int): Target depth for the images.
        window_level (float): Window level for image windowing.
        window_width (float): Window width for image windowing.
    """
    output_image_dir = Path(output_image_dir)
    output_segment_dir = Path(output_segment_dir)

    output_image_dir.mkdir(parents=True, exist_ok=True)
    output_segment_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_file)

    for _, row in df.iterrows():
        img_path = Path(row["volume_path"])
        seg_path = Path(row["segment_path"])

        processed_image, processed_segment, affine = process_image_and_segment(
            img_path, seg_path, target_depth, window_level, window_width
        )

        save_processed_files(
            processed_image,
            processed_segment,
            affine,
            img_path,
            seg_path,
            output_image_dir,
            output_segment_dir,
        )
