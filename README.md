# Diffusion Model with U-Net Architecture

## Overview
This project implements a **Diffusion Model** using a **U-Net Architecture** to generate high-quality images from noise. The model leverages forward and reverse diffusion processes to train and optimize image generation. The approach involves adding noise to images over timesteps and training the U-Net to predict and remove this noise during reverse diffusion.

The repository includes a detailed implementation of:
- **Forward Diffusion** (adding noise to images progressively)
- **U-Net Architecture** (for noise prediction and denoising)
- **Reverse Diffusion** (removing noise step-by-step to generate an image)
- **Variance Loss** and **Noise Prediction Loss**
- Training and inference pipelines

## Features
- **Customizable U-Net Model:** Supports skip connections and multiple down-sampling/up-sampling blocks.
- **Noise Scheduling:** Uses a scheduler to control the rate of noise addition during forward diffusion.
- **Training Metrics:** Includes loss monitoring and optimizer setup.
- **Inference Pipeline:** Converts noise into meaningful images using reverse diffusion.

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Setup and Installation](#setup-and-installation)
- [Model Architecture](#model-architecture)
  - [Forward Diffusion](#forward-diffusion)
  - [U-Net Architecture](#u-net-architecture)
  - [Reverse Diffusion](#reverse-diffusion)
- [Training Details](#training-details)
- [Results](#results)


## Setup and Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/diffusion-unet
   cd diffusion-unet
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Ensure you have a GPU-enabled environment for efficient training.

## Model Architecture

### Forward Diffusion
In forward diffusion, noise is added progressively to the image over a series of timesteps using a **scheduler**. This process converts the image into a Gaussian noise distribution. The noise scheduling controls the variance of the added noise at each step.

**Key Code:**
```python
noisy_image = noise_scheduler.add_noise(image, noise, timestep)
```

### U-Net Architecture
The U-Net is designed to predict and remove noise added to the images during the forward diffusion process. The network consists of:
- **DownBlocks:** For extracting multi-scale features.
- **UpBlocks:** For reconstructing the image with skip connections.
- **Conditioning:** The model is conditioned on timestep and textual embeddings.

**Key Code:**
```python
output = unet(noisy_image, timestep_embedding, text_embedding)
```

### Reverse Diffusion
During reverse diffusion, the U-Net iteratively removes noise step-by-step based on its predictions until a clean image is generated.

**Key Code:**
```python
predicted_noise = unet(noisy_image, timestep, conditional_embedding)
denormalized_image = reverse_diffusion(noisy_image, predicted_noise, timestep)
```

## Training Details
- **Input:** A dataset of images.
- **Loss Function:**
  - **Noise Prediction Loss:** Measures the difference between the predicted and actual noise.
  - **Variance Loss:** Captures uncertainty in the noise prediction.
- **Optimizer:** Adam optimizer for adjusting the model weights.
- **Training Loop:** At each timestep:
  - Noise is added to the image.
  - The model predicts the added noise.
  - The loss is computed and backpropagated.

**Key Code:**
```python
loss = noise_loss(predicted_noise, actual_noise) + variance_loss
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

## Results
- The model demonstrates the ability to reconstruct images effectively from noise.
- Loss decreases consistently during training, indicating model optimization.




