import torch.nn as nn
from plate_settings import NUM_CLASSES

class PlateDetectorANN(nn.Module):

    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.4), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.3), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.2), nn.Linear(128, 2))

    def forward(self, x):
        return self.network(x)

class OCRNet(nn.Module):

    def __init__(self, input_dim, num_classes=NUM_CLASSES):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, 512), nn.BatchNorm1d(512), nn.SiLU(), nn.Dropout(0.4), nn.Linear(512, 256), nn.BatchNorm1d(256), nn.SiLU(), nn.Dropout(0.35), nn.Linear(256, 128), nn.BatchNorm1d(128), nn.SiLU(), nn.Dropout(0.3), nn.Linear(128, 64), nn.BatchNorm1d(64), nn.SiLU(), nn.Dropout(0.2), nn.Linear(64, num_classes))

    def forward(self, x):
        return self.net(x)