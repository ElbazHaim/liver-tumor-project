from pathlib import Path
from tqdm import tqdm
import numpy as np
from collections import Counter
import nibabel as nib
import json

data_root_dir = Path("/mnt/data/haim_asaf/data")
raw_data_path = data_root_dir / "raw"

raw_volumes = raw_data_path / "volumes"
raw_segs = raw_data_path / "segmentations"

num_volumes = len(list(raw_volumes.iterdir()))
print(f"Found {num_volumes} volumes")


def load_nifti_image(path) -> np.array:
    """Loads the nifti image as a 3D numpy array"""
    return nib.load(str(path)).get_fdata()


def count_class_pixels(mask_files):
    patient_class_counts = Counter()
    slice_class_counts = Counter()
    labels = ["background_overall", "liver", "tumor"]

    # Iterate over patients
    for mask_file in tqdm(mask_files):
        mask = load_nifti_image(mask_file)

        unique = np.unique(mask)
        patient_class_counts.update(dict(zip(labels, [1 for _ in range(len(unique))])))

        # Within each patient, iterate over slices
        for slice_ in tqdm(mask):
            unique_slice = np.unique(slice_)
            slice_class_counts.update(
                dict(zip(labels, [1 for _ in range(len(unique_slice))]))
            )
            slice_class_counts.update(
                {
                    "background_exclusive": 1
                    if 0 in unique_slice and len(unique_slice) == 1
                    else 0
                }
            )

    return {
        "patient_class_counts": patient_class_counts,
        "slice_class_counts": slice_class_counts,
    }


segmentations = list(raw_segs.iterdir())
class_distribution: dict = count_class_pixels(segmentations)
print(class_distribution)

with open("class_distribution.json", "w") as f:
    json.dump(class_distribution, f)
