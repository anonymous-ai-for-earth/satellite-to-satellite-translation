import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_ch, 128, 3, padding=1, stride=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, 1, padding=0, stride=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, out_ch, 3, padding=1, stride=1),
        )

    def forward(self, x):
        return self.model(x)

class Decoder(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(Decoder, self).__init__()
        self.model = nn.Sequential(
                nn.Conv2d(in_ch, 128, 3, padding=1, stride=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, 128, 1, padding=0, stride=1),
                nn.LeakyReLU(0.2),
                nn.Conv2d(128, out_ch, 3, padding=1, stride=1),
        )

    def forward(self, x):
        return self.model(x)

class AutoEncoder(nn.Module):
    def __init__(self, in_ch, enc_dim):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(in_ch, enc_dim)
        self.decoder = Decoder(enc_dim, in_ch)

    def forward(self, x):
        enc = self.encoder(x)
        x_dec = self.decoder(enc)
        return x_dec
