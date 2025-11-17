import torch
import torch.nn as nn
import torch.nn.functional as F

class Trainer:
    def __init__(self, model_name:str, model, optimizer, criterion, save_pth:str,
                 num_epochs:int=100, device:str="cuda:0", seed:int=42, model_pth:str=None,
                 model_encoder=None, early_stopping_dice:float=0.93):
        torch.cuda.manual_seed(seed)
        self.num_epochs = num_epochs
        self.optimizer = optimizer
        self.criterion = criterion
        self.model_name = model_name
        self.model = model
        self.model_encoder = model_encoder
        if model_pth is not None:
            self.model.load_state_dict(torch.load(model_pth, map_location=device))
        self.device = device
        self.save_pth = save_pth
        self.log_interval = 30
        self.early_stopping_dice = early_stopping_dice

        # Lists to store training and validation metrics
        self.train_losses = []
        self.val_losses = []
        self.train_dices = []
        self.val_dices = []
        self.train_ious = []
        self.val_ious = []

        # Best model and its metrics
        self.best_model = None
        self.best_dice = 0.0
        self.best_epoch = 0

    def dice_coeff(self, predicted, target, smooth=1e-5):
        intersection = torch.sum(predicted * target)
        union = torch.sum(predicted) + torch.sum(target)
        dice = (2. * intersection + smooth) / (union + smooth)
        return dice
    
    def iou(self, pred_mask, true_mask):
        intersection = torch.logical_and(pred_mask, true_mask).sum().item()
        union = torch.logical_or(pred_mask, true_mask).sum().item()
        iou_score = intersection / union if union != 0 else 0.0
        return iou_score

    def save_best_model(self, epoch, dice):
        if dice > self.best_dice:
            self.best_dice = dice
            self.best_epoch = epoch
            torch.save(self.model.state_dict(), self.save_pth)

    def train(self, dataloader):
        train_loader, val_loader = dataloader["train"], dataloader["test"]
        for epoch in range(self.num_epochs):
            train_loss = 0.0
            train_dice = 0.0
            train_iou = 0.0
            val_loss = 0.0
            val_dice = 0.0
            val_iou = 0.0
            maxpooler = nn.MaxPool2d(kernel_size=16, stride=16)

            # Training loop
            for i, (images, masks) in enumerate(train_loader):
                images, masks = images.to(self.device), masks.to(self.device)
                
                self.model.train()
                self.optimizer.zero_grad()

                
                if self.model_name == "TEUnet":
                    outputs = self.model(images)
                    loss = (self.criterion(outputs[0], maxpooler(masks)) + self.criterion(outputs[1], masks)) / 2
                    outputs = outputs[1]
                elif self.model_name == "TEUnet2Encoder":
                    outputs = self.model(images)[0]
                    masks = maxpooler(masks)
                    loss = self.criterion(outputs, masks)
                elif self.model_name == "TEUnet2Decoder":
                    outputs = self.model(self.model_encoder(images))
                    loss = self.criterion(outputs, masks)
                elif self.model_name in ["Unet++", "Segformer", "DeepLabV3+", "DeepLabV3", "SwinUNETR"]:
                    outputs = self.model(images)
                    outputs = F.sigmoid(outputs)
                    loss = self.criterion(outputs, masks)
                else:
                    outputs = self.model(images)
                    loss = self.criterion(outputs, masks)
                loss.backward()
                self.optimizer.step()

                train_loss += loss.item()
                train_dice += self.dice_coeff(outputs>0.5, masks)
                train_iou += self.iou(outputs>0.5, masks)

            # Validation loop
            self.model.eval()
            with torch.no_grad():
                for images, masks in val_loader:
                    images, masks = images.to(self.device), masks.to(self.device)
                    if self.model_name == "TEUnet":
                        outputs = self.model(images)[1]
                    elif self.model_name == "TEUnet2Encoder":
                        outputs = self.model(images)[0]
                        masks = maxpooler(masks)
                    elif self.model_name == "TEUnet2Decoder":
                        outputs = self.model(self.model_encoder(images))
                    elif self.model_name in ["Unet++", "Segformer", "DeepLabV3+", "DeepLabV3", "SwinUNETR"]:
                        outputs = F.sigmoid(self.model(images))
                    else:
                        outputs = self.model(images)
                    
                    val_loss += self.criterion(outputs, masks).item()
                    val_dice += self.dice_coeff(outputs>0.5, masks)
                    val_iou += self.iou(outputs>0.5, masks)

            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            avg_train_dice = train_dice / len(train_loader)
            avg_val_dice = val_dice / len(val_loader)
            avg_train_iou = train_iou / len(train_loader)
            avg_val_iou = val_iou / len(val_loader)

            #print(f'Epoch [{epoch + 1}/{self.num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
            print(f'Epoch [{epoch+1:3d}/{self.num_epochs}], Train Dice: {avg_train_dice:.4f}, Val Dice: {avg_val_dice:.4f}, Train IoU: {avg_train_iou:.4f}, Val IoU: {avg_val_iou:.4f}')

            # Save metrics
            self.train_losses.append(avg_train_loss)
            self.val_losses.append(avg_val_loss)
            self.train_dices.append(avg_train_dice)
            self.val_dices.append(avg_val_dice)
            self.train_ious.append(avg_train_iou)
            self.val_ious.append(avg_val_iou)

            # Save best model
            self.save_best_model(epoch + 1, avg_val_dice)
            if avg_train_dice >= self.early_stopping_dice:
                print(f"Early stopping at epoch {epoch+1} with Val Dice: {avg_val_dice:.4f}")
                break
    
    def get_metrics(self):
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_dices': self.train_dices,
            'val_dices': self.val_dices,
            'train_ious': self.train_ious,
            'val_ious': self.val_ious,
            'best_model': self.best_model,
            'best_dice': self.best_dice,
            'best_epoch': self.best_epoch
        }

