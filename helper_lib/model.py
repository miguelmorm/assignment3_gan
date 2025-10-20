import torch
import torch.nn as nn

# ---------- Generator (exact spec) ----------
class Generator(nn.Module):
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, 7 * 7 * 128),
            nn.BatchNorm1d(7 * 7 * 128),
            nn.ReLU(True),
        )
        self.deconv = nn.Sequential(
            # 128x7x7 -> 64x14x14
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            # 64x14x14 -> 1x28x28
            nn.ConvTranspose2d(64, 1, kernel_size=4, stride=2, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        x = self.net(z)                 # (B, 7*7*128)
        x = x.view(-1, 128, 7, 7)       # (B, 128, 7, 7)
        return self.deconv(x)           # (B, 1, 28, 28)

# ---------- Discriminator (exact spec) ----------
class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            # 1x28x28 -> 64x14x14
            nn.Conv2d(1, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            # 64x14x14 -> 128x7x7
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.fc = nn.Linear(128 * 7 * 7, 1)  # real/fake logit

    def forward(self, x):
        h = self.conv(x)
        h = h.view(h.size(0), -1)
        return self.fc(h)

def get_model(model_name: str, **kwargs):
    """
    Extendable helper as in Module 4.
    For Part 1: return GAN dict with 'G' and 'D'.
    """
    name = (model_name or "").upper()
    if name == "GAN":
        return {"G": Generator(kwargs.get("z_dim", 100)), "D": Discriminator()}
    raise ValueError(f"Unknown model_name: {model_name}")
