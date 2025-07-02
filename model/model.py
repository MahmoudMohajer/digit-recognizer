import torch.nn as nn 
import torch

class DigitCNN(nn.Module): 
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3),  # 28x28 → 26x26
            nn.ReLU(),
            nn.MaxPool2d(2),                 # 26x26 → 13x13
            nn.Conv2d(32, 64, kernel_size=3),# 13x13 → 11x11
            nn.ReLU()
        )

        # Dynamically determine flattened size
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, 28, 28)
            out = self.features(dummy_input)
            self.flat_dim = out.view(1, -1).size(1)  # e.g., 64 * 11 * 11 = 7744

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(self.flat_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self. classifier(x)
        return x
