import torch.nn as nn

class BetterCNN(nn.Module):
    def __init__(self, num_classes=27):
        super().__init__()

        def conv_block(in_c, out_c):
            return nn.Sequential(
                nn.Conv2d(in_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.Conv2d(out_c, out_c, 3, padding=1),
                nn.BatchNorm2d(out_c),
                nn.ReLU(),
                nn.MaxPool2d(2)
            )

        self.features = nn.Sequential(
            conv_block(3, 32),      # 224 → 112
            conv_block(32, 64),     # 112 → 56
            conv_block(64, 128),    # 56 → 28
            conv_block(128, 256),   # 28 → 14
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 14 * 14, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x
