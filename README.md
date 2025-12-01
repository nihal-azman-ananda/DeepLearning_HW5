# DDPM on CIFAR-10: Standard & Advanced Architectures

This repository contains a PyTorch implementation of **Denoising Diffusion Probabilistic Models (DDPM)** applied to the CIFAR-10 dataset. It explores the generative capabilities of diffusion models through two distinct architectures: a standard U-Net baseline and an advanced variant incorporating Self-Attention and Exponential Moving Average (EMA).

## 📄 Project Overview

The goal of this project is to generate high-fidelity 64x64 images by learning the reverse diffusion process. The models are trained to iteratively denoise isotropic Gaussian noise into coherent images.

### Models Implemented
1.  **Standard DDPM:** A simplified U-Net architecture focusing on core diffusion mechanics.
2.  **Advanced DDPM:** An enhanced U-Net featuring Multi-Head Self-Attention mechanisms, deeper residual blocks, and EMA weight averaging for improved stability and sample quality.

## 🧠 Model Architectures

### 1. Standard DDPM (`DDPM.ipynb`)
* **Backbone:** Simple U-Net with convolutional downsampling and upsampling.
* **Resolution:** 64x64x3.
* **Diffusion Schedule:** Linear $\beta$ schedule ($10^{-4}$ to $0.02$) over $T=1000$ timesteps.
* **Training:** Optimized using Adam ($lr=3 \times 10^{-4}$) with simple MSE loss between predicted and actual noise.
* **Tech Stack:** Uses Automatic Mixed Precision (AMP) and Gradient Clipping for stability.

### 2. Advanced DDPM (`DDPM_adv.ipynb`)
* **Backbone:** Deep U-Net with "DoubleConv" (Conv-GroupNorm-SiLU) blocks.
* **Attention Mechanisms:** Integrated **Multi-Head Self-Attention** at 16x16 and 8x8 resolutions to capture global dependencies.
* **Stabilization:** Implements **Exponential Moving Average (EMA)** to smooth model weights during inference.
* **Training:** Optimized using Adam ($lr=2 \times 10^{-4}$) with a longer training horizon.

## 📊 Experimental Results

Both models were evaluated using the **Frechet Inception Distance (FID)** metric. 

| Model | Epochs Trained | Best FID (approx.) | Status |
| :--- | :--- | :--- | :--- |
| **Standard DDPM** | 60 | **114.8** | Completed |
| **Advanced DDPM** | 180 | **100.2** | Stopped Early* |

*\*Note: The Advanced DDPM training was terminated at epoch 180 (out of 350) due to resource constraints on the compute cluster. Despite incomplete training, it significantly outperformed the standard model, demonstrating the effectiveness of attention mechanisms and EMA.*

### Key Observations
* **Architecture Matters:** The addition of attention blocks at lower resolutions allowed the Advanced model to generate sharper edges and more coherent global structures compared to the Standard model.
* **Convergence:** The Advanced model showed a consistent downward trend in FID, suggesting it would have achieved significantly lower scores had training completed.

## 🚀 Usage

### Requirements
* Python 3.8+
* PyTorch (CUDA supported)
* Torchvision
* Numpy, Matplotlib

### Running the Models
The project is structured into Jupyter Notebooks for ease of visualization and experimentation.

1.  **Clone the repository:**
    ```bash
    git clone
    ```

2.  **Train the Standard Model:**
    Open `DDPM.ipynb`. This notebook handles data loading for CIFAR-10, defines the simple U-Net, and runs the training loop with periodic FID evaluation.

3.  **Train the Advanced Model:**
    Open `DDPM_adv.ipynb`. This notebook contains the complex U-Net definition (with Attention and DoubleConv), the EMA helper class, and the advanced training loop.

## 📉 Evaluation Metric: FID
Evaluation is performed using the **Frechet Inception Distance (FID)**. The code extracts feature embeddings from a pre-trained Inception-v3 network to compare the statistics of generated images against real CIFAR-10 samples.

*FID calculation code is included in the notebooks/helper files.*

## 📚 References
* Ho, J., Jain, A., & Abbeel, P. (2020). *Denoising Diffusion Probabilistic Models*. NeurIPS.
* CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html