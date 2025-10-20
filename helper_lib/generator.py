import os, io, base64, torch
from torchvision.utils import make_grid, save_image
from PIL import Image
from .model import get_model

@torch.no_grad()
def generate_samples(
    G_ckpt_path="./checkpoints/G_latest.pt",
    num_samples=16,
    z_dim=100,
    device="cpu",
    out_path="./outputs/generated_grid.png",
):
    device = torch.device(device if torch.cuda.is_available() or device == "cpu" else "cpu")
    G = get_model("GAN", z_dim=z_dim)["G"].to(device)
    G.load_state_dict(torch.load(G_ckpt_path, map_location=device)); G.eval()

    z = torch.randn(num_samples, z_dim, device=device)
    imgs = G(z).cpu()
    grid = make_grid(imgs, nrow=int(num_samples**0.5), normalize=True, value_range=(-1,1))

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    save_image(grid, out_path)

    # also return base64 (handy for API response)
    arr = (grid.permute(1,2,0).numpy() * 255).clip(0,255).astype("uint8")
    buf = io.BytesIO(); Image.fromarray(arr).save(buf, format="PNG")
    return {"image_base64": base64.b64encode(buf.getvalue()).decode("utf-8"),
            "out_path": out_path}
