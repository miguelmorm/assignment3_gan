from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
import os
import torch
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples, generate_mnist_samples
from helper_lib.model import get_model

app = FastAPI(title="SPS GAN API", version="1.0.0")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
Z_DIM = 100

# ---------- Schemas ----------
class TrainRequest(BaseModel):
    epochs: int = 1
    batch_size: int = 128
    z_dim: int = 100
    lr: float = 0.0002
    betas0: float = 0.5
    betas1: float = 0.999
    device: str = "cpu"

class GenerateRequest(BaseModel):
    G_ckpt_path: Optional[str] = "./checkpoints/G_latest.pt"
    num_samples: int = 16
    z_dim: int = 100
    device: str = "cpu"

# ---------- Health ----------
@app.get("/health")
def health():
    return {"status": "ok"}

# ---------- Part 1 ----------
@app.post("/train_gan")
def train(req: TrainRequest):
    return train_gan(
        epochs=req.epochs,
        batch_size=req.batch_size,
        z_dim=req.z_dim,
        lr=req.lr,
        betas=(req.betas0, req.betas1),
        device=req.device,
        out_dir="./outputs",
        ckpt_dir="./checkpoints",
    )

@app.post("/generate_gan")
def generate(req: GenerateRequest):
    return generate_samples(
        G_ckpt_path=req.G_ckpt_path,
        num_samples=req.num_samples,
        z_dim=req.z_dim,
        device=req.device,
        out_path="./outputs/generated_grid.png",
    )

# ---------- Part 2 (MNIST) ----------
@app.post("/train_gan_mnist")
def train_gan_mnist(epochs: int = 1, batch_size: int = 128, device: str = DEVICE):
    """
    Reuses your existing train_gan() which already trains on MNIST.
    Saves checkpoints under ./checkpoints and images under ./outputs.
    """
    result = train_gan(
        epochs=epochs,
        batch_size=batch_size,
        z_dim=Z_DIM,
        lr=2e-4,
        betas=(0.5, 0.999),
        device=device,
        out_dir="./outputs",
        ckpt_dir="./checkpoints",
    )
    return {"message": "MNIST GAN training complete", **result}

@app.post("/generate_mnist")
def generate_mnist(num_samples: int = 25, device: str = DEVICE, seed: Optional[int] = None):
    """
    Loads the latest MNIST generator weights if present and writes a 5x5 grid PNG.
    """
    models = get_model("GAN", z_dim=Z_DIM)
    G = models["G"]
    ckpt = "./checkpoints/G_latest.pt"
    if os.path.exists(ckpt):
        G.load_state_dict(torch.load(ckpt, map_location=device))
    out_path = generate_mnist_samples(
        G, device=device, num_samples=num_samples, z_dim=Z_DIM,
        out_path="outputs/mnist_generated.png", seed=seed
    )
    return {"generated_image": out_path}
