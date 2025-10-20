import os, time
from typing import Dict, List
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.utils import make_grid, save_image
from .model import get_model

def _weights_init(m):
    name = m.__class__.__name__
    if "Conv" in name or "Linear" in name:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
        if getattr(m, "bias", None) is not None:
            nn.init.constant_(m.bias.data, 0)

def get_mnist_loader(batch_size=128, train=True):
    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),  # -> [-1, 1]
    ])
    ds = datasets.MNIST(root="./data", train=train, transform=tfm, download=True)
    return DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=2, pin_memory=True)

@torch.no_grad()
def _sample_images(G, device, n=16, z_dim=100):
    z = torch.randn(n, z_dim, device=device)
    G.eval()
    imgs = G(z).cpu()
    G.train()
    return imgs

def train_gan(
    epochs=1,
    batch_size=128,
    z_dim=100,
    lr=2e-4,
    betas=(0.5, 0.999),
    device="cpu",
    out_dir="./outputs",
    ckpt_dir="./checkpoints",
) -> Dict:
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    models = get_model("GAN", z_dim=z_dim)
    G, D = models["G"].to(device), models["D"].to(device)
    G.apply(_weights_init); D.apply(_weights_init)

    # --- Standard GAN losses (as in class, not WGAN) ---
    criterion = nn.BCEWithLogitsLoss()
    opt_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
    opt_D = optim.Adam(D.parameters(), lr=lr, betas=betas)

    loader = get_mnist_loader(batch_size=batch_size, train=True)
    os.makedirs(out_dir, exist_ok=True); os.makedirs(ckpt_dir, exist_ok=True)

    G_losses: List[float] = []; D_losses: List[float] = []
    real_label = 0.9   # label smoothing for real
    fake_label = 0.0

    for epoch in range(1, epochs + 1):
        g_acc, d_acc, nb = 0.0, 0.0, 0
        t0 = time.time()

        for real, _ in loader:
            nb += 1
            real = real.to(device)
            bsz = real.size(0)

            # ---- Train D ----
            D.zero_grad(set_to_none=True)
            logits_real = D(real)
            loss_real = criterion(logits_real, torch.full((bsz,1), real_label, device=device))

            z = torch.randn(bsz, z_dim, device=device)
            fake = G(z).detach()
            logits_fake = D(fake)
            loss_fake = criterion(logits_fake, torch.full((bsz,1), fake_label, device=device))

            loss_D = loss_real + loss_fake
            loss_D.backward(); opt_D.step()

            # ---- Train G ----
            G.zero_grad(set_to_none=True)
            z = torch.randn(bsz, z_dim, device=device)
            gen = G(z)
            logits_gen = D(gen)
            loss_G = criterion(logits_gen, torch.full((bsz,1), 1.0, device=device))
            loss_G.backward(); opt_G.step()

            d_acc += loss_D.item(); g_acc += loss_G.item()

        G_losses.append(g_acc/nb); D_losses.append(d_acc/nb)

        # sample grid each epoch
        with torch.no_grad():
            imgs = _sample_images(G, device, n=16, z_dim=z_dim)
        grid = make_grid(imgs, nrow=4, normalize=True, value_range=(-1,1))
        save_image(grid, os.path.join(out_dir, f"samples_epoch_{epoch:03d}.png"))

        # checkpoints
        torch.save(G.state_dict(), os.path.join(ckpt_dir, f"G_epoch_{epoch}.pt"))
        torch.save(D.state_dict(), os.path.join(ckpt_dir, f"D_epoch_{epoch}.pt"))

        print(f"[Epoch {epoch}/{epochs}] G: {G_losses[-1]:.4f} | D: {D_losses[-1]:.4f} | {time.time()-t0:.1f}s")

    # latest pointers
    torch.save(G.state_dict(), os.path.join(ckpt_dir, "G_latest.pt"))
    torch.save(D.state_dict(), os.path.join(ckpt_dir, "D_latest.pt"))

    return {
        "G_losses": G_losses,
        "D_losses": D_losses,
        "G_ckpt": os.path.join(ckpt_dir, "G_latest.pt"),
        "D_ckpt": os.path.join(ckpt_dir, "D_latest.pt"),
        "last_grid": os.path.join(out_dir, f"samples_epoch_{epochs:03d}.png"),
    }
