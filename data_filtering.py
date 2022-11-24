import os
import cv2
import numpy as np
import shutil
import pydicom
from tqdm import tqdm
from pathlib import Path


if __name__ == "__main__":
    images_dir = r"F:\Piotrek\mri_dataset\mri_results"
    results_dir = r"F:\Piotrek\mri_dataset\mri_results_filtered"
    
    Path(results_dir).mkdir(exist_ok=True, parents=True)

    for subdir, dirs, files in tqdm(os.walk(images_dir), desc="Walking through files..."):
        for file in files:
            filepath = os.path.join(subdir, file)
            
            # TODO: check resolution and save original and downsampled 
