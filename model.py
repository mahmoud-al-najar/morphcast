from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # 16, 20, 20
            nn.ReLU(True),
            nn.MaxPool2d(2),  # 16, 10, 10
            nn.Conv2d(16, 8, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d(2),   # 8, 5, 5
        )
        self.decoder = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(8, 16, kernel_size=1, stride=1),  # 16, 10, 10
            nn.ReLU(True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
            nn.Conv2d(16, 1, kernel_size=1, stride=1),  # 1, 20, 20
            nn.ReLU(True),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
