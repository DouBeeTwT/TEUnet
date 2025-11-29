import torch
from torch.optim import Adam
from Scripts.utils.data import Database2Dataloader
from Scripts.utils.trainer import Trainer
from Scripts.model.loss import bce_dice_loss, focal_dice_loss
import warnings
warnings.filterwarnings("ignore")
import argparse

parser = argparse.ArgumentParser(description='')
parser.add_argument('-n', '--num_epochs', default=100, type=int, help='number of training epochs')
parser.add_argument('-lr', '--learning_rate', default=1e-4, type=float, help='learning rate for training')
parser.add_argument('-d', '--device', default="cuda:0", type=str, help='device to use for training')
parser.add_argument('-b', '--batch_size', default=16, type=int, help='batch size for training')
parser.add_argument('-s', '--seed', default=[0,1,2,3,4], type=int, nargs="+", help='random seed for training')
parser.add_argument('-dn','--database_name', default=["BUSI"], type=str, nargs="+", help='database name for training')
parser.add_argument('-m', '--model_name', default=["Unet", "AttUnet", "Unet++", "DeepLabV3", "AAUnet", "SwinUNETR"], type=str, nargs="+", help='model name for training')
parser.add_argument('--image_size', default=512, type=int, help='image size for training')
parser.add_argument('--in_channels', default=1, type=int, help='input channels')
parser.add_argument('--out_channels', default=1, type=int, help='output channels')
parser.add_argument('--hidden_channels', default=64, type=int, help='hidden channels')
parser.add_argument('-p', '--drop_out', default=0.0, type=float, help='drop out rate')
parser.add_argument('-es', '--early_stopping_dice', default=0.93, type=float, help='early stopping dice score')
parser.add_argument('--retrain', action='store_true', help='whether to retrain the model from scratch')
args = parser.parse_args()

device=args.device
num_epochs = args.num_epochs
seed_list = args.seed
batch_size = args.batch_size
image_size = args.image_size
database_name_list = args.database_name
model_name_list = args.model_name
in_channels = args.in_channels
out_channels = args.out_channels
hidden_channels = args.hidden_channels
p = args.drop_out
need_retrain = args.retrain
#model_name_list = ["Unet", "AttUnet", "Unet++","Segformer", "DeepLabV3", "SwinUNETR", "AAUnet", "TEUnet2Encoder", "TEUnet2Decoder"]

# Data
for database_name in database_name_list:
    for seed in seed_list:
        print("Loading database:", database_name, " Seed:", seed)
        dataloader = Database2Dataloader(
            database_path=f"./Database/{database_name}",
            image_size=image_size,
            batch_size=batch_size,
            seed=seed)
        print("Database loaded.")

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
                model.encoder.load_state_dict(torch.load(f"Checkpoints/{database_name}/{model_name}_{feature_size}_{seed}.pth", map_location=device))
                feature_size = hidden_channels//2
            elif model_name == "TEUnet2Encoder":
                from Scripts.model import TEUnet2_Encoder
                model = TEUnet2_Encoder(in_channels,out_channels,hidden_channels//2, p)
                feature_size = hidden_channels//2
            elif model_name == "TEUnet2Decoder":
                feature_size = hidden_channels//2
                from Scripts.model import TEUnet2_Encoder, TEUnet2_Decoder
                model_encoder = TEUnet2_Encoder(in_channels,out_channels,hidden_channels//2, p)
                model_encoder.load_state_dict(torch.load(f"Checkpoints/{database_name}/TEUnet2Encoder_{feature_size}_{seed}.pth", map_location=device))
                model_encoder.to(device).eval()
                model = TEUnet2_Decoder(in_channels,out_channels,hidden_channels//2, p)  
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
            model.to(device)
            # Trainedr

            model_pth = f"Checkpoints/{database_name}/{model_name}_{feature_size}_{seed}.pth" if need_retrain else None
            trainer = Trainer(model_name=model_name, model=model, num_epochs=num_epochs, device=device, seed=seed, model_pth = model_pth,
                            optimizer=Adam(model.parameters(), lr=args.learning_rate, weight_decay=1e-6),
                            criterion=focal_dice_loss, early_stopping_dice=args.early_stopping_dice,
                            save_pth=f"./Checkpoints/{database_name}/{model_name}_{feature_size}_{seed}.pth",
                            model_encoder= model_encoder if model_name=="TEUnet2Decoder" else None)

            trainer.train(dataloader)