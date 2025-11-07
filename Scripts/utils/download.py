import kagglehub
import os

os.environ['KAGGLEHUB_CACHE'] = 'Database'
path = kagglehub.dataset_download("aryashah2k/breast-ultrasound-images-dataset")