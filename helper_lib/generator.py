# helper_lib/generator.py
import os
import io
import base64
import torch
from torchvision.utils import make_grid, save_image
from PIL import Image
from .model import get_model

# === Part 1 helper (kept): generate grid and return base64 + file ===
@torch.no_grad()
def generate_samples(
    G_ckpt_path: str = "./checkpoints/G_latest.pt",
    num_samples: int = 16,
    z_dim: int = 100,
    device: str = "cpu",
    out_path: str = "./outputs/generated_grid.png",
):
    """
    Loads a trained Generator checkpoint and creates a grid image.
    This is the original Part 1 helper you already had; we keep it
    for backwards compatibility with /generate_gan.
    """
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    models = get_model("GAN", z_dim=z_dim)
    G = models["G"].to(device)
    if os.path.exists(G_ckpt_path):
        G.load_state_dict(torch.load(G_ckpt_path, map_location=device))
    G.eval()

    z = torch.randn(num_samples, z_dim, device=device)
    imgs = G(z).cpu()
    grid = make_grid(imgs, nrow=int(num_samples**0.5), normalize=True, value_range=(-1, 1))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path)

    ndarr = (grid.permute(1, 2, 0).numpy() * 255).clip(0, 255).astype("uint8")
    im = Image.fromarray(ndarr)
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
    return {"image_base64": b64, "out_path": out_path}

# === Part 2 helper (new): generate MNIST grid to file ===
@torch.no_grad()
def generate_mnist_samples(
    model_G,
    *,
    device: str = "cpu",
    num_samples: int = 25,
    z_dim: int = 100,
    out_path: str = "outputs/mnist_generated.png",
    seed: int | None = None,
):
    """
    Uses an in-memory Generator (optionally loaded from checkpoints by caller)
    to generate num_samples MNIST images and saves a 5x5 grid PNG.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    if seed is not None:
        torch.manual_seed(seed)

    model_G = model_G.to(device).eval()
    z = torch.randn(num_samples, z_dim, device=device)
    imgs = model_G(z).cpu()
    save_image(imgs, out_path, nrow=5, normalize=True, value_range=(-1, 1))
    return out_path
