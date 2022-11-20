import os
import cv2
import numpy as np
import shutil
import pydicom
from tqdm import tqdm
import nibabel as nib
from pathlib import Path


def num_gen():
    num = 0
    while True:
        yield num
        num += 1


def check_quality(img, threshold=0.95):
    return (np.count_nonzero([img < 10]) / (img.shape[0] * img.shape[1])) < threshold # and np.mean(img[0:5, 0:5]) < 0.1


def save_dicom(filepath: str, outpath: str):
    ds = pydicom.read_file(filepath)
    img = 255 * cv2.normalize(ds.pixel_array, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    
    if len(img.shape) > 2:
        for plane_nr in range(img.shape[0]):
            img_slice = img[plane_nr, :, :]

            if check_quality(img_slice):
                cv2.imwrite(os.path.join(outpath, str(next(number_gen)) + '.png'), img_slice.astype(np.uint8))
    else:
        if check_quality(img):
            cv2.imwrite(os.path.join(outpath, str(next(number_gen)) + '.png'), img.astype(np.uint8))
    

def save_png(filepath: str, outpath: str):
    shutil.copyfile(filepath, os.path.join(outpath, str(next(number_gen)) + '.png'))
    
    
def save_nii(filepath: str, outpath: str):
    scan = nib.load(filepath)
    scan = scan.get_fdata()
    
    for plane in range(scan.shape[2]):
        img = 255 * cv2.normalize(scan[:, :, plane], None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        if check_quality(img):
            cv2.imwrite(os.path.join(outpath, str(next(number_gen)) + '.png'), img.astype(np.uint8))


if __name__ == "__main__":
    rootdir = r"F:\Piotrek\mri_data"
    results_dir = r"F:\Piotrek\mri_results"
    number_gen = num_gen()

    Path(results_dir).mkdir(exist_ok=True, parents=True)

    for subdir, dirs, files in tqdm(os.walk(rootdir), desc="Walking through files..."):
        for file in files:
            filepath = os.path.join(subdir, file)
            file_name, file_extension = os.path.splitext(file)
            try:
                if file_extension == ".png" or file_extension == ".jpg" or file_extension == ".jpeg":
                    save_png(filepath, results_dir)
                elif file_extension == ".nii":
                    save_nii(filepath, results_dir)
                elif file_extension == ".dcm":
                    save_dicom(filepath, results_dir)
                else:
                    continue
            except Exception as e:
                print("Ups...", e)
            
            

            
        