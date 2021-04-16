from torch import nn


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 8, kernel_size=3),
            nn.ReLU(True),
            nn.MaxPool2d(2)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 2, stride=2),
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 2, stride=2, padding=2),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
