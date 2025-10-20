from fastapi import FastAPI
from pydantic import BaseModel
from typing import Optional
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples

app = FastAPI(title="SPS GAN API", version="1.0.0")

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

@app.get("/health")
def health():
    return {"status": "ok"}

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
