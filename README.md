# UMR-GAN: Image Restoration / Translation on 4-Class Brain MRI

UMR-GAN is a **mask-conditioned conditional GAN** (cGAN) for **denoising** and **inpainting** brain MRI slices. The project includes a Colab-ready setup, a modular `src/` training pipeline (PyTorch), and a Gradio UI for interactive inference.

---

## Table of Contents
- [Project Purpose](#project-purpose)
- [Repository Structure](#repository-structure)
- [Quick Start (Colab)](#quick-start-colab)
- [Local Setup](#local-setup)
- [Training & Evaluation](#training--evaluation)
- [Launch the Gradio Interface](#launch-the-gradio-interface)
- [Current Results](#current-results)
- [Known Issues & Troubleshooting](#known-issues--troubleshooting)
- [Reproducibility Checklist](#reproducibility-checklist)
- [Citations](#citations)
- [Author](#author)

---

## Project Purpose
Restore MRI images that are corrupted by noise or missing regions so downstream tasks (clinical review, segmentation, radiomics) receive higher-quality inputs. We use a U-Net generator + multi-scale PatchGAN discriminator with a loss that mixes **mask-weighted L1**, **adversarial**, **SSIM**, and **perceptual** terms.

---

## Repository Structure

```text
├── data/                    # cleaned/processed data (4 class folders)
│   ├── training
│   └── test
├── notebooks/               # setup & UI demo notebooks
│   ├── training.ipynb
│   ├── test.ipynb
│   └── setup.ipynb
├── src/                     # main scripts (data, models, training)
├── ui/                      # Gradio/Streamlit/Flask app files (interface)
│   └── app.ipynb
├── results/                 # checkpoints, samples, plots
└── docs/                    # architecture diagrams, UI screenshots

---

# UMR-GAN: Image Restoration / Translation on 4-Class Brain MRI

UMR-GAN is a **mask-conditioned conditional GAN** (cGAN) for **denoising** and **inpainting** brain MRI slices. The project includes a Colab-ready setup, a modular `src/` training pipeline (PyTorch), and a Gradio UI for interactive inference.

---

## Project Purpose
Restore MRI images corrupted by noise or missing regions so downstream tasks (clinical review, segmentation, radiomics) receive higher-quality inputs. We use a U-Net generator + multi-scale PatchGAN discriminator with a loss that mixes **mask-weighted L1**, **adversarial**, **SSIM**, and **perceptual** terms.

---

## Quick Start (Colab)

1. Upload your dataset to Google Drive at `My Drive/training/` (4 class folders).
2. Open **Colab** and run one of:
   - `notebooks/setup_drive_only.ipynb` (recommended; uses your Drive folder directly)
   - `notebooks/setup_colab.ipynb` (Drive link via `gdown` or ZIP)
   - `notebooks/setup.ipynb` (simple Drive path variant)
3. If needed, set:
   ```python
   from pathlib import Path
   DATA_DIR = Path('/content/drive/MyDrive/training') ("https://drive.google.com/drive/folders/1VjGdzJbmKK14s2qK3ijMeiKtD6wJ6Fns?usp=sharing")
```text
4. Run all cells. You should see:
    GPU/environment check
    Class counts & corrupt-file scan
    Random image grid and size distribution plots
    DataLoader smoke test

---

## Local Setup
Colab is easiest. For local runs with Python 3.12:

# Clone
git clone https://github.com/sanjanakodavali/UMR_GAN.git
cd UMR_GAN

# Create environment
conda create -n umr-gan python=3.12 -y
conda activate umr-gan

# Install minimal CPU deps
pip install torch torchvision pillow matplotlib pandas jupyter nbformat pyyaml

# (Optional) Install CUDA build of torch for your system:
# https://pytorch.org/get-started/locally/

