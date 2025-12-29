import numpy as np
import torch
from Scripts.utils.data import Database2Dataloader
from tqdm import tqdm
import torch.nn.functional as F
import csv

class metrics():
    def __init__(self,
                 csv_path:str,
                 device:str="cuda:0",
                 smooth:float=1e-5):
        self.csv_path = csv_path
        self.device = device
        self.smooth = smooth

    def calc_score(self, predicted, target):
        # Add count
        n = predicted.shape[0]
        self.score["n"] += n
        # Dice
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        self.score["dice"] += dice.item()*n
        # IoU
        if torch.sum(predicted) == 0 and torch.sum(target) == 0:
            self.score["iou"] += 1.0
        else:
            union = torch.logical_or(predicted, target).sum()
            iou = intersection / union if union != 0 else torch.tensor(0.0)
            self.score["iou"] += iou.item()*n
        # Recall, Precision, Specificity
        true_positive = torch.sum(predicted * target)
        false_positive = torch.sum(predicted * (1-target))
        true_negtive = torch.sum((~predicted) * (1 - target))
        false_negative = torch.sum(target * (~predicted))
        recall = true_positive / (true_positive + false_negative + self.smooth)
        self.score["recall"] += recall.item()*n
        precision = true_positive / (true_positive + false_positive + self.smooth)
        self.score["precision"] += precision.item()*n
        specificity = true_negtive / (true_negtive + false_positive + self.smooth)
        self.score["specificity"] += specificity.item()*n
    
    def load_dataset(self, database_name:str, batch_size:int=1, seed:int=0):
        self.database_name = database_name.split('-')[-1]
        dataloader = Database2Dataloader(database_path=f"Database/{database_name}", batch_size=batch_size, seed=seed)
        self.dataloader = dataloader["test"]

    def load_model(self, model_name:str, in_channels:int, out_channels:int, hidden_channels:int, seed:int, p:float=0.0, show_attention=False):
        if model_name == "Unet":
            from Scripts.model import Unet
            model = Unet(in_channels,out_channels,hidden_channels, p)
            feature_size = hidden_channels
        elif model_name == "AttUnet":
            from Scripts.model import AttUnet
            model = AttUnet(in_channels,out_channels,hidden_channels, p)
            feature_size = hidden_channels
        elif model_name == "AAUnet":
            from Scripts.model import AAUnet
            model = AAUnet(in_channels,out_channels,hidden_channels//2, p)
            feature_size = hidden_channels//2
        elif model_name == "TEUnet":
            from Scripts.model import TEUnet
            model = TEUnet(in_channels,out_channels,hidden_channels//2, p)
            feature_size = hidden_channels//2
        elif model_name == "TEUnet2":
            feature_size = hidden_channels//2
            from Scripts.model import TEUnet2_Encoder, TEUnet2_Decoder, TEUnet2
            model = TEUnet2(in_channels,out_channels,hidden_channels//2, p, show_attention=show_attention)
            model.encoder.load_state_dict(torch.load(f"./Checkpoints/{self.database_name}/TEUnet2Encoder_{feature_size}_{seed}.pth", map_location=self.device))
            model.decoder.load_state_dict(torch.load(f"./Checkpoints/{self.database_name}/TEUnet2Decoder_{feature_size}_{seed}.pth", map_location=self.device))
            model.to(self.device).eval()
            self.model_name = model_name
            self.model = model
            self.pth = f"./Checkpoints/{self.database_name}/{model_name}_{feature_size}_{seed}.pth"
        elif model_name == "Unet++":
            from Scripts.model import UnetPlusPlus
            model = UnetPlusPlus(in_channels, out_channels, hidden_channels)
            feature_size = hidden_channels
        elif model_name == "Segformer":
            from segmentation_models_pytorch import Segformer
            model = Segformer(encoder_name="resnet34", in_channels=in_channels, classes=out_channels)
            feature_size = 64
        elif model_name == "DeepLabV3+":
            from segmentation_models_pytorch import DeepLabV3Plus
            model = DeepLabV3Plus(encoder_name="resnet34", in_channels=in_channels, classes=out_channels)
            feature_size = 64
        elif model_name == "DeepLabV3":
            from segmentation_models_pytorch import DeepLabV3
            model = DeepLabV3(encoder_name="resnet34", in_channels=in_channels, classes=out_channels)
            feature_size = 64
        elif model_name == "SwinUNETR":
            from monai.networks.nets import SwinUNETR
            model = SwinUNETR(in_channels=in_channels, out_channels=out_channels, spatial_dims=2, feature_size=hidden_channels//4*3)
            feature_size = hidden_channels//4*3
        else:
            raise NotImplementedError(f"{model_name} is not implemented")

        if model_name not in ["TEUnet2"]:
            model.load_state_dict(torch.load(f"./Checkpoints/{self.database_name}/{model_name}_{feature_size}_{seed}.pth", map_location=self.device))
            model.to(self.device).eval()
            self.model_name = model_name
            self.model = model
            self.pth = f"./Checkpoints/{self.database_name}/{model_name}_{feature_size}_{seed}.pth"
    
    def save(self):
        self.score["pth"] = self.pth
        del self.score["n"]
        with open(self.csv_path, mode='a', newline='') as file:
            writer = csv.DictWriter(file, fieldnames=["pth","dice","iou","recall","precision","specificity"])
            if file.tell() == 0:
                writer.writeheader()
            writer.writerow(self.score)

    def evaluate(self, threshold:float=0.5):
        self.score = {
            "n": 0,
            "dice": 0.0,
            "iou": 0.0,
            "recall": 0.0,
            "precision": 0.0,
            "specificity": 0.0}
        with torch.no_grad():
            for images, masks in tqdm(self.dataloader, leave=False, ncols=50):
                images, masks = images.to(self.device), masks.to(self.device)
                if self.model_name == "TEUnet":
                    outputs = self.model(images)[1]
                elif self.model_name == "TEUnet2":
                    outputs = self.model_decoder(self.model_encoder(images))
                elif self.model_name in ["Unet++", "Segformer", "DeepLabV3+", "DeepLabV3"]:
                    outputs = self.model(images)
                    outputs = F.sigmoid(outputs)
                else:
                    outputs = self.model(images)

                self.calc_score(outputs>threshold, masks)
        
        # Average the scores
        for key in self.score.keys():
            if key != "n":
                self.score[key] /= self.score["n"]
                self.score[key] = round(self.score[key], 4)
        
        # save to csv
        self.save()