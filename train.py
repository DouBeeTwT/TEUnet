from torch.optim import Adam
from Scripts.utils.data import Database2Dataloader
from Scripts.utils.trainer import Trainer
from Scripts.model.loss import bce_dice_loss

device="cuda:2"
num_epochs = 100
seed_list = [0]
batch_size = 4
image_size = 512
database_name_list = ["BUSI"]
model_name_list = ["TEUnet"]
in_channels = 1
out_channels = 1
hidden_channels = 64
p = 0.05

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
                model.to(device)
            elif model_name == "AttUnet":
                from Scripts.model import AttUnet
                model = AttUnet(in_channels,out_channels,hidden_channels, p)
                model.to(device)
            elif model_name == "AAUnet":
                from Scripts.model import AAUnet
                model = AAUnet(in_channels,out_channels,hidden_channels//2, p)
                model.to(device)
            elif model_name == "TEUnet":
                from Scripts.model import TEUnet
                model = TEUnet(in_channels,out_channels,hidden_channels, p)
                model.to(device)

            # Trainer
            trainer = Trainer(model_name=model_name, model=model, num_epochs=num_epochs, device=device, seed=seed,
                            optimizer=Adam(model.parameters(), lr=1e-4, weight_decay=1e-6),
                            criterion=bce_dice_loss,
                            pth=f"./Checkpoints/{model_name}_{database_name}_{seed}.pth")

            trainer.train(dataloader)