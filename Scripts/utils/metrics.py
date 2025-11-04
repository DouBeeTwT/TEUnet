import numpy as np

class metrics():
    def __init__(self,
                 model,
                 dataloader,
                 device:str="cuda:0",
                 theholds:float=0.5,
                 smooth:float=1e-10):
        self.model = model.to(device)
        self.dataloader = dataloader
        self.device = device
        self.theholds = theholds
        self.smooth = smooth
        self.scores = {
            "jeccard":0.0,
            "precision":0.0,
            "recall":0.0,
            "specificity":0.0,
            "dice":0.0}
     
    def to_numpy(self, x):
        return x.detach().cpu().numpy()

    def calc(self):
        n = 0
        for images, groundtruth in self.dataloader:
            prediction = self.model(images.to(self.device)) > self.theholds
            prediction, groundtruth = self.to_numpy(prediction), self.to_numpy(groundtruth)

            tp = np.sum((prediction == 1) & (groundtruth == 1))
            tn = np.sum((prediction == 0) & (groundtruth == 0))
            fp = np.sum((prediction == 1) & (groundtruth == 0))
            fn = np.sum((prediction == 0) & (groundtruth == 1))
            union = np.sum((prediction == 1) | (groundtruth == 1))
            total = np.sum(prediction == 1) + np.sum(groundtruth == 1)

            self.scores["jeccard"] += tp / (union + self.smooth)
            self.scores["precision"] += tp / (tp + fp + self.smooth)
            self.scores["recall"] += tp / (tp + fn + self.smooth)
            self.scores["specificity"] += tn / (tn + fp + self.smooth)
            self.scores["dice"] += (2 * tp) / (total + self.smooth)
            n += images.shape[0]

        for key in self.scores.keys:
            self.scores[key] /= n
        
        return self.scores