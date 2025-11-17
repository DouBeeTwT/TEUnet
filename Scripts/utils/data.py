import numpy as np
import os
import cv2
from torch.utils.data import Dataset, DataLoader
import torch
import glob
import pandas as pd
import albumentations as A
from sklearn.model_selection import train_test_split

class CustomImageMaskDataset(Dataset):
    def __init__(self, dataframe, transform, seed:int=42):
        self.data = dataframe
        self.transform = transform
        np.random.seed(seed)
        torch.manual_seed(seed)
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path = self.data.iloc[idx]['image_path']
        mask_path = self.data.iloc[idx]['mask_path']

        image = cv2.imread(image_path, 0)
        mask = cv2.imread(mask_path, 0)

        if self.transform:
            transformed = self.transform(image=np.array(image), mask=np.array(mask))
            image, mask = transformed["image"] / 255.0, transformed["mask"].unsqueeze(0) / 255
        
        return image, mask
    

def Database2Dataloader(database_path:str = "./Database/Dataset_BUSI_with_GT",
                        image_size:int = 512,
                        batch_size:int = 16,
                        seed:int = 42) -> dict:
    dataloader = {"train":None, "test":None}
    if not os.path.exists(database_path):
        print("Can't find the database pathway. Please Download {} first.")
        return dataloader
    masks = glob.glob(f"{database_path}/*/*_mask.png")
    images = [mask_images.replace("_mask", "") for mask_images in masks]
    series = list(zip(images, masks))
    dataset = pd.DataFrame(series, columns=['image_path', 'mask_path'])
    train, test= train_test_split(dataset, test_size=0.25, random_state=seed, shuffle=True)

    transform = A.Compose([
        A.Normalize(0.330, 0.221),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, p=1.0),
        A.Rotate(limit=(-10,10), p=0.5),
        A.RandomBrightnessContrast(brightness_limit=(-0.05,0.05), contrast_limit=(-0.05,0.05), p=0.3),
        A.RandomScale(scale_limit=(-0.1, 0.1), p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Resize(image_size, image_size),
        A.ToTensorV2(),
    ])
    dataloader["train"] = DataLoader(CustomImageMaskDataset(train, transform, seed), batch_size, True, drop_last=True)

    transform = A.Compose([
        A.Normalize(0.330, 0.221),
        A.PadIfNeeded(min_height=image_size, min_width=image_size, border_mode=0, p=1.0),
        A.Resize(image_size, image_size),
        A.ToTensorV2(),
    ])
    dataloader["test"] = DataLoader(CustomImageMaskDataset(test, transform, seed), batch_size)

    return dataloader