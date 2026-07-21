# ProjectsDL

A comprehensive collection of Deep Learning projects focusing on **generative models** and **image processing**, implemented from scratch with detailed understanding of underlying concepts.

## 📋 Philosophy

> *Everything is coded by hand for learning purposes. I prioritize understanding over copying, and only use AI for visualization.*

All implementations are built without copying code from online repositories or research papers. The goal is deep comprehension of modern generative modeling techniques, with a focus on:
- Core algorithmic understanding
- Custom architecture implementations
- Experimental variations and ablations
- Visual analysis of model behavior

---

## 🎯 Projects Overview

### 1. **Diffusion Models** 📁 `Diffusion/`

Complete implementation of diffusion-based generative models trained on MNIST with advanced techniques.

**Key Notebooks:**
- **`ddpm.ipynb`** - Basic Diffusion Models (DDPM/DDIM)
  - Forward diffusion process
  - Reverse diffusion sampling
  - Various network architectures (U-Net, Transformers)
  - Loss functions and training strategies

- **`latentdiffusion.ipynb`** - Latent Space Diffusion
  - VAE encoder/decoder for latent space compression
  - Diffusion in compressed latent representation
  - Comparison of latent vs pixel-space diffusion

- **`freeguidance.ipynb`** - Conditional Generation
  - Classifier-free guidance implementation
  - Conditional generation techniques
  - Guidance scale effects on generation quality

- **`dit.ipynb`** - Diffusion Transformers
  - Transformer-based noise prediction
  - Positional embeddings and attention mechanisms
  - DiT architecture variants
  - Performance comparisons with U-Net

**Architectures Explored:**
- U-Net with skip connections
- Vision Transformers (ViT)
- Hybrid architectures combining CNNs and Transformers

---

### 2. **Flow Models** 📁 `flowmodels/`

Implementation of flow-based generative models with focus on recent advances.

**Key Notebooks:**
- **`babyreflow.ipynb`** - Rectified Flow
  - Flow matching for generative modeling
  - Faster sampling with optimal transport
  - Trajectory analysis and visualization
  - Comparison with traditional diffusion sampling

---

### 3. **Research Papers Implementation** 📁 `codingpapers/`

Deep implementations of recent research papers with experimental analysis.

**Key Notebooks:**
- **`Latent1.ipynb`** - VAE & Latent Space Analysis
  - Variational Autoencoder from scratch
  - Latent space interpolation analysis
  - VAE training dynamics
  - Visual interpolation between encoded samples

- **`restormer.ipynb`** - Image Restoration
  - Restormer architecture implementation
  - Multi-scale feature restoration
  - Image enhancement pipeline

**Included Visualizations:**
- `latentinterpolation.png` - Smooth transitions in latent space
- `outputvaeflux.png` - VAE generation samples
- `rgbinterpolation.png` - RGB space interpolation effects
- `plotcosinelatvspix.png` - Latent vs Pixel space cosine similarity analysis
- `plotcosinefluxvae2.png` - Flow model cosine analysis

---

### 4. **Perceptual Loss** 📁 `perceptualoss/`

Deep dive into perceptual loss functions and their applications in neural networks.

**Key Notebooks:**
- **`perceptual_loss.ipynb`** - Perceptual Loss Implementation
  - Feature extraction from pre-trained networks
  - Perceptual distance metrics
  - Optimization using perceptual gradients
  - Analysis of neuron activation patterns

**Included Resources:**
- `2602.04193v1.pdf` - Research paper reference
- `blurredneuron.jpg` - Blurred neuron activation visualization
- `eachneuron.jpg` - Individual neuron response visualization

---

### 5. **RAW to sRGB Pipeline** 📁 `rawtosrgb/`

Image processing project implementing RAW format conversion to sRGB color space.

**Key Notebooks:**
- **`pipelinersrgb.ipynb`** - RAW Image Processing
  - Demosaicing algorithms
  - Color correction and white balance
  - Gamma correction
  - RAW to sRGB conversion pipeline

**Included Files:**
- `rawboat.ARW` - Sample Sony RAW image file
- `out_srgb.png` - Processed output example

---

## 📊 Visual Results

### Sample Outputs

The `photos/` directory contains training results and generated samples:

**Diffusion Model Outputs:**
- `outddpmc.png`, `outddpmc5d.png` - DDPM generation samples
- `outputddim.png` - DDIM sampler results
- `outputddpmim.png` - DDPM with various configurations

**Advanced Techniques:**
- `1explatentdiffusion.png` - Latent diffusion outputs
- `dittransformer.png` - Diffusion Transformer architecture
- `trajectoriesreflow.png` - Rectified flow trajectories

**Conditional Generation:**
- `outputcfg.png` - Classifier-free guidance results
- `output7d.png`, `output6d.png` - Multi-dimensional generation samples

**Architecture Analysis:**
- `experience_orthogposembeddim64.png` - Positional embedding analysis
- `losswithdiscri.png` - Loss curves with discriminator

---

## 🔬 Technical Topics Covered

### Generative Modeling
- ✅ Diffusion models (DDPM, DDIM)
- ✅ Latent diffusion
- ✅ Flow matching & Rectified flows
- ✅ Variational Autoencoders (VAE)
- ✅ Transformers for generation

### Loss Functions & Training
- ✅ Perceptual losses
- ✅ Adversarial training
- ✅ Classifier-free guidance
- ✅ Custom optimization strategies

### Deep Learning Techniques
- ✅ U-Net architectures
- ✅ Vision Transformers (ViT)
- ✅ Positional embeddings & variants
- ✅ Attention mechanisms
- ✅ Skip connections & residual paths

### Image Processing
- ✅ RAW format handling & demosaicing
- ✅ Color space conversions
- ✅ Image restoration & enhancement
- ✅ Interpolation in latent spaces

---

## 🚀 Future Work

- [ ] Restormer (image restoration)
- [ ] NafNet (fast image restoration)
- [ ] Escape PyTorch black boxes (`torch.cuda.amp.GradScaler()`, `torch.compiler()`)
- [ ] Single flow model per technique:
  - [ ] Distillation-based flow
  - [ ] Self-distillation from scratch
  - [ ] Marginal-from-conditional learning
- [ ] Deep understanding of:
  - [ ] Forward automatic differentiation
  - [ ] Reverse automatic differentiation

---

## 📚 Learning Resources Used

This repository demonstrates implementation of concepts from:
- DDPM/DDIM diffusion models
- Latent Diffusion Models
- Vision Transformers
- Rectified Flows & Flow Matching
- Perceptual Loss Functions
- VAE principles
- Image restoration techniques

---

## 💻 Technical Stack

- **Framework:** PyTorch
- **Notebooks:** Jupyter Notebook
- **Visualization:** Matplotlib, PIL
- **Image Processing:** OpenCV, PIL
- **Numerical Computing:** NumPy, Scipy

---

## 📝 Notes

- All code is implementation-focused with extensive comments explaining concepts
- Results are visualized and compared across different configurations
- Each project includes architectural diagrams and mathematical foundations
- No external model weights or pre-built solutions copied from online sources

---

**Created:** May 2026  
**Status:** Active Development
