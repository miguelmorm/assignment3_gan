# Assignment 3 – GAN (PyTorch) + FastAPI + Docker

This project implements a **Generative Adversarial Network (GAN)** using **PyTorch**, served via a **FastAPI** web application, and containerized using **Docker**.  
It includes endpoints to train the GAN and generate synthetic images, as well as a health check and auto-generated API documentation.

---

## Project Overview

**Frameworks:** PyTorch, FastAPI, Uvicorn  
**Containerization:** Docker  
**Environment:** Python 3.10+  
**Author:** Miguel Morales  
**Course:** Applied Generative AI – Columbia University  

---

## Project Structure

ASSIGNMENT3_GAN/  
├── app/ → FastAPI entry point (defines endpoints)  
├── helper_lib/ → Model, training loop, and image generation utilities  
├── checkpoints/ → Saved model weights  
├── outputs/ → Generated samples  
├── data/ → Dataset folder  
├── requirements.txt → Dependencies  
├── Dockerfile → Docker configuration  
├── .gitignore → Ignored files and folders  
└── README.md → Project documentation  

---

## Setup Instructions

**Local installation**

pip install -r requirements.txt

**Run with Docker**

docker build -t sps-gan:latest .
docker run --rm -p 8000:8000 sps-gan:latest

Access the app at **http://127.0.0.1:8000**.

---

## API Endpoints

| Endpoint | Method | Description |
|-----------|---------|-------------|
| `/health` | GET | Check server health |
| `/train_gan` | POST | Train the GAN model (1-epoch example) |
| `/generate_gan` | POST | Generate synthetic images using the trained GAN |
| `/docs` | GET | Swagger UI interactive docs |
| `/redoc` | GET | Alternative API documentation view |

---

## Example Requests

Train the GAN:
curl -X POST http://127.0.0.1:8000/train_gan

Generate images:
curl -X POST http://127.0.0.1:8000/generate_gan

Check health:
curl http://127.0.0.1:8000/health

---

## Example Logs

INFO: Application startup complete.
INFO: 127.0.0.1:51208 - "GET /health HTTP/1.1" 200 OK
Epoch [1/1] G: 2.7809 | D: 0.6010 | Time: 77.0s
INFO: 127.0.0.1:51232 - "POST /train_gan HTTP/1.1" 200 OK
INFO: 127.0.0.1:51258 - "POST /generate_gan HTTP/1.1" 200 OK

---

## Part 2 – Model Deployment with MNIST (GAN Integration)

This section expands the GAN to train on the **MNIST dataset** and integrates it with the FastAPI endpoints.

### Objective
Train a GAN capable of generating realistic hand-written digits using the MNIST dataset and make it accessible via the API endpoints.

### Technical Implementation
- **Dataset:** MNIST (60,000 hand-written digits, 28×28 grayscale)  
- **Architecture:** Generator and Discriminator trained adversarially  
- **Training:** BCE loss + Adam optimizers in PyTorch  
- **Endpoints Integrated:**  
  - `/train_gan` – trains the model on MNIST  
  - `/generate_gan` – generates hand-written digits  
- **Helper Library Enhancements:**  
  - `model.py` – defines Generator and Discriminator via `get_model()`  
  - `trainer.py` – implements training loop and saves outputs  
  - `generator.py` – handles sample image generation

---

### How It Works
1. POST `/train_gan` → trains model and saves checkpoints under `/checkpoints`  
2. POST `/generate_gan` → loads weights and saves grid at `/outputs/generated_grid.png`  
3. Normalizes output images to [−1, 1] for display

---

### Results and Evaluation
- GAN successfully generates realistic MNIST-style digits.  
- Demonstrates an end-to-end machine learning pipeline: **training → inference → API deployment → Docker containerization**.  
- The app can be tested via Swagger UI at:  
  **http://127.0.0.1:8000/docs**

---

## Testing Instructions
1. Run `uvicorn app.main:app --reload`  
2. Open `http://127.0.0.1:8000/docs`  
3. Interact with `/train_gan`, `/generate_gan`, and `/health`

---

**Summary:**  
This project combines deep learning, API design, and containerization into a single deployable solution demonstrating a complete lifecycle for Generative AI model integration.
