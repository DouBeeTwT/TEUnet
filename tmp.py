import glob
import os
from tqdm import tqdm

image_files = glob.glob("Database/HAM10000/images/*.jpg")
# jpg转png
from matplotlib.pyplot import imsave, imread
for image_file in tqdm(image_files, ncols=80, leave=False):
    image = imread(image_file)
    imsave(image_file.replace(".jpg", ".png"), image)
    os.remove(image_file)