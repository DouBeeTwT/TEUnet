import torch
from torch.optim import Adam
from Scripts.utils.data import Database2Dataloader
from Scripts.utils.trainer import Trainer
from Scripts.model.loss import bce_dice_loss
import warnings
warnings.filterwarnings("ignore")

device="cuda:3"
num_epochs = 200
seed_list = [1]
batch_size = 4
image_size = 512
database_name_list = ["BUSI"]
#model_name_list = ["Unet", "AttUnet", "AAUnet", "TEUnet2Encoder", "TEUnet2Decoder","Unet++","Segformer", "DeepLabV3", "SwinUNETR"]
model_name_list = ["SwinUNETR"]
in_channels = 1
out_channels = 1
hidden_channels = 64
p = 0.0
need_retrain = False

# Data
for database_name in database_name_list:
    for seed in seed_list:
        dataloader = Database2Dataloader(
            database_path=f"./Database/{database_name}",
            image_size=image_size,
            batch_size=batch_size,
            seed=seed)

        # Model
        for model_name in model_name_list:
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
                model.encoder.load_state_dict(torch.load(f"Checkpoints/{model_name}_{database_name}_{seed}.pth", map_location=device))
                feature_size = hidden_channels//2
            elif model_name == "TEUnet2Encoder":
                from Scripts.model import TEUnet2_Encoder
                model = TEUnet2_Encoder(in_channels,out_channels,hidden_channels//4, p)
                feature_size = hidden_channels//4
            elif model_name == "TEUnet2Decoder":
                feature_size = hidden_channels//4
                from Scripts.model import TEUnet2_Encoder, TEUnet2_Decoder
                model_encoder = TEUnet2_Encoder(in_channels,out_channels,hidden_channels//4, p)
                model_encoder.load_state_dict(torch.load(f"Checkpoints/TEUnet2Encoder_{feature_size}_{database_name}_{seed}.pth", map_location=device))
                model_encoder.to(device).eval()
                model = TEUnet2_Decoder(in_channels,out_channels,hidden_channels//4, p)  
            elif model_name == "Unet++":
                from segmentation_models_pytorch import UnetPlusPlus
                model = UnetPlusPlus(encoder_name="resnet34", in_channels=in_channels, classes=out_channels)
                feature_size = 64
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
            model.to(device)
            # Trainedr

            model_pth = f"Checkpoints/{model_name}_{feature_size}_{database_name}_{seed}.pth" if need_retrain else None
            trainer = Trainer(model_name=model_name, model=model, num_epochs=num_epochs, device=device, seed=seed, model_pth = model_pth,
                            optimizer=Adam(model.parameters(), lr=1e-4, weight_decay=1e-6),
                            criterion=bce_dice_loss,
                            save_pth=f"./Checkpoints/{model_name}_{feature_size}_{database_name}_{seed}.pth",
                            model_encoder= model_encoder if model_name=="TEUnet2Decoder" else None)

            trainer.train(dataloader)