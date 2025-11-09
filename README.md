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
├── notebooks/               # setup & UI demo notebooks
│   ├── setup_drive_only.ipynb
│   ├── setup_colab.ipynb
│   └── setup.ipynb
├── src/                     # main scripts (data, models, training)
│   ├── config/
│   │   ├── base.yaml
│   │   ├── train_denoise.yaml
│   │   └── train_inpaint.yaml
│   ├── data/                # MRISliceDataset + DataLoader
│   ├── models/              # UNetG, PatchGAN D, losses, factory
│   ├── training/            # loop, optimizers, checkpoints
│   ├── metrics/             # PSNR/SSIM (placeholder)
│   ├── utils/               # config loader, seeding
│   └── cli/
│       └── train.py
├── ui/                      # Gradio/Streamlit/Flask app files (interface)
├── results/                 # checkpoints, samples, plots
└── docs/                    # architecture diagrams, UI screenshots
