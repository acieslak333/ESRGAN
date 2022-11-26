import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt


def downsample_image(img, fig, ax, interpolation_method='quadric', name='test', size=448): 
    my_dpi = int(size/8) 
    ax.imshow(img, interpolation=interpolation_method, cmap='gray')
    
    if not os.path.isfile(name): 
        fig.savefig(name, dpi=my_dpi) 
    else: 
        fig.savefig(name, dpi=my_dpi) 
    
    ax.clear()


if __name__ == "__main__":
    images_dir = r"F:\Piotrek\mri_dataset\mri_results"
    original_dir = r"F:\Piotrek\mri_dataset\mri_results_filtered\original"
    downsampled_dir = r"F:\Piotrek\mri_dataset\mri_results_filtered\downsampled"
    
    Path(original_dir).mkdir(exist_ok=True, parents=True)
    Path(downsampled_dir).mkdir(exist_ok=True, parents=True)
    
    f = plt.figure(frameon=False) 
    f.set_size_inches(8, 8) 
    ax = plt.Axes(f, [0., 0., 1., 1.]) 
    ax.set_axis_off() 
    f.add_axes(ax) 

    for file in tqdm(os.listdir(images_dir), desc="Downsampling files..."):
        filepath = os.path.join(images_dir, file)
        im = Image.open(filepath)
        width, height = im.size
        pix = np.array(im)

        if width == height and width > 250 and width % 4 == 0 and len(np.unique(pix)) > 2:
            shutil.copyfile(filepath, os.path.join(original_dir, file))
            downsample_image(im, f, ax, name=os.path.join(downsampled_dir, file), size=width/4)

