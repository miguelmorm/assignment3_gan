Assignment 3 – GAN (PyTorch) + FastAPI + Docker

This project implements a Generative Adversarial Network (GAN) using PyTorch, served via a FastAPI web application, and containerized using Docker.
It includes endpoints to train the GAN and generate synthetic images, as well as a health check and auto-generated API documentation.
Project Overview
Frameworks: PyTorch, FastAPI, Uvicorn
Containerization: Docker
Environment: Python 3.10+
Author: Miguel Morales
Course: Applied Generative AI – Columbia University
Project Structure
ASSIGNMENT3_GAN/
│
├── app/
│   └── main.py          # FastAPI entry point (defines endpoints)
│
├── helper_lib/
│   ├── model.py         # Generator and Discriminator classes
│   ├── trainer.py       # GAN training loop
│   └── generator.py     # Image generation utilities
│
├── checkpoints/         # Saved model weights
├── outputs/             # Generated samples
├── data/                # Dataset folder
│
├── requirements.txt     # Dependencies
├── Dockerfile           # Docker configuration
├── .gitignore           # Ignored files and folders
└── README.md            # Project documentation

Requirements
Install dependencies locally (recommended inside a virtual environment)
pip install -r requirements.txt
Alternatively, build and run with Docker
# Build the image
docker build -t sps-gan:latest .

# Run the container
docker run --rm -p 8000:8000 sps-gan:latest
This command runs the FastAPI app inside a Docker container and exposes it at port 8000.
API Endpoints
Endpoint	Method	Description
/health	GET	Check server health
/train_gan	POST	Train the GAN model (1-epoch example)
/generate_gan	POST	Generate synthetic images using the trained GAN
/docs	GET	Interactive Swagger UI documentation
/redoc	GET	Alternative API documentation view
Example Requests (with curl)
Train the GAN
curl -X POST http://127.0.0.1:8000/train_gan
Generate new images
curl -X POST http://127.0.0.1:8000/generate_gan
Check health
curl http://127.0.0.1:8000/health
Example Logs (Training)
INFO: Application startup complete.
INFO: 127.0.0.1:51208 - "GET /health HTTP/1.1" 200 OK
Epoch [1/1] G: 2.7809 | D: 0.6010 | Time: 77.0s
INFO: 127.0.0.1:51232 - "POST /train_gan HTTP/1.1" 200 OK
INFO: 127.0.0.1:51258 - "POST /generate_gan HTTP/1.1" 200 OK
Part 2 – Model Deployment with MNIST (GAN Integration)
In Part 2 of this assignment, the GAN model was expanded and integrated with the MNIST dataset to demonstrate the full pipeline of training, generation, and deployment through FastAPI.
Objective
The goal was to train a GAN capable of generating realistic hand-written digits using the MNIST dataset, and to make it accessible through the existing API endpoints.
This part also required extending the helper library modules developed in previous modules to support the new training and generation workflow.
Technical Implementation
Dataset: MNIST (60,000 hand-written digit images – 28×28 grayscale)
Architecture: Two neural networks – Generator and Discriminator – trained adversarially.
Training: Implemented in PyTorch with BCE loss and Adam optimizers.
Endpoints Integrated:
/train_gan – Trains the GAN on MNIST for a user-defined number of epochs.
/generate_gan – Generates synthetic hand-written digits using the trained generator.
Helper Library Enhancements:
model.py – Defines and returns the Generator and Discriminator through get_model().
trainer.py – Implements train_gan() to load MNIST, train the model, and save checkpoints and sample grids.
generator.py – Adds functions to generate and save images from latent noise vectors.
How It Works
Train the model – Send a POST request to /train_gan.
The API downloads the MNIST dataset, trains both networks, and saves checkpoints under /checkpoints.
Generate images – Send a POST request to /generate_gan.
This loads the latest Generator weights and produces a grid of synthetic digits at /outputs/generated_grid.png.
Visualization – All images are normalized to [–1, 1] and saved as 28×28 grayscale samples for evaluation.
Expected Outcome
After training, the GAN produces realistic MNIST-style digits that can be visualized via the generated grid.
The Generator successfully learns to map random noise vectors to digit-like patterns through adversarial learning.
Extended API Endpoints
Endpoint	Method	Description
/train_gan	POST	Train the MNIST-based GAN and save checkpoints
/generate_gan	POST	Generate synthetic MNIST digit samples
/health	GET	Server status check
/docs	GET	Swagger UI documentation
/redoc	GET	Alternative API view
Evaluation Notes
Demonstrates integration of deep learning training and deployment through FastAPI.
Meets Module 6 objectives by linking GAN training to API endpoints.
Includes data loading, model definition, training loop, and inference workflow, all via HTTP requests.
Represents a complete end-to-end workflow from model design to deployment and Docker execution.
