# Assignment 3 – GAN (PyTorch) + FastAPI + Docker

This project implements a Generative Adversarial Network (GAN) using **PyTorch**, served via a **FastAPI** web application, and containerized using **Docker**.  
It includes endpoints to train the GAN and generate synthetic images, as well as a health check and auto-generated API documentation.

---

## 🚀 Project Overview
- **Frameworks:** PyTorch, FastAPI, Uvicorn  
- **Containerization:** Docker  
- **Environment:** Python 3.10+  
- **Author:** Miguel Morales  
- **Course:** Applied Generative AI – Columbia University

---

## 📁 Project Structure
ASSIGNMENT3_GAN/
│
├── app/
│ ├── main.py # FastAPI entry point (defines endpoints)
│ ├── schemas.py # Pydantic models
│
├── helper_lib/
│ ├── generator.py # GAN Generator class
│ ├── trainer.py # GAN training loop
│
├── checkpoints/ # Saved model weights
├── outputs/ # Generated samples
├── data/ # Dataset folder
│
├── requirements.txt # Dependencies
├── Dockerfile # Docker configuration
├── .gitignore # Ignored files and folders
└── README.md # Project documentation

---

##Requirements

-----Install dependencies locally (recommended inside a virtual environment):
pip install -r requirements.txt
Alternatively, build and run with Docker (see below).
-----Docker Setup
1. Build the Docker image
docker build -t sps-gan:latest .
2. Run the container
docker run --rm -p 8000:8000 sps-gan:latest
This command runs the FastAPI app inside a Docker container and exposes it at port 8000.
-----API Endpoints
Endpoint	Method	Description
/health	GET	Check server health
/train_gan	POST	Train the GAN model (1 epoch example)
/generate_gan	POST	Generate synthetic images using the trained GAN
/docs	GET	Interactive Swagger UI documentation
/redoc	GET	Alternative API documentation view
-----Example Requests (with curl)
Train the GAN:
curl -X POST http://127.0.0.1:8000/train_gan
Generate new images:
curl -X POST http://127.0.0.1:8000/generate_gan
Check health:
curl http://127.0.0.1:8000/health
-----Example Logs (Training)
INFO: Application startup complete.
INFO: 127.0.0.1:51208 - "GET /health HTTP/1.1" 200 OK
Epoch [1/1] G: 2.7809 | D: 0.6010 | Time: 77.0s
INFO: 127.0.0.1:51232 - "POST /train_gan HTTP/1.1" 200 OK
INFO: 127.0.0.1:51258 - "POST /generate_gan HTTP/1.1" 200 OK
