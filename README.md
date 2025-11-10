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
```
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

4. Run all cells. You should see:
    - GPU/environment check
    - Class counts & corrupt-file scan
    - Random image grid and size distribution plots
    - DataLoader smoke test

---
```text
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
```
---
## Training & Evaluation

**Train (denoise):**
```bash
python -m src.cli.train --cfg src/config/train_denoise.yaml
```
**Train (inpaint):**
```bash
python -m src.cli.train --cfg src/config/train_inpaint.yaml
```
- Checkpoints/metrics are written to runs/exp_<hash>/ and results/.
- Validation reports mean PSNR and SSIM each epoch.
---
## Launch the Gradio Interface

### Colab notebook UI
Open your UI notebook (e.g., `ui/app.ipynb`) and run all:
- Upload or pick a sample slice
- Choose **denoise** or **inpaint**
- Click **Restore**, preview, and download

### Local Python app
```bash
pip install gradio
python ui/app.py
# then open the printed URL (e.g., http://127.0.0.1:7860)
```
## Current Results
- **UMR-GAN (held-out mean):** PSNR **34.231 dB**, SSIM **0.918**
- **Baseline (Noisy→Clean):** PSNR **30.191 dB**, SSIM **0.749**
- **Gain:** **+4.04 dB** PSNR, **+0.169** SSIM

**Qualitative triplets** are in `results/`:
- `triplet_denoise_2_Te-glTr_0001.png`, `triplet_denoise_3_Te-glTr_0002.png`
- `triplet_inpaint_1_Te-glTr_0000.png`, `triplet_inpaint_3_Te-glTr_0002.png`

---

## Known Issues & Troubleshooting
- **Dataset shape:** requires **4** class subfolders. If different, update paths/class discovery.
- **DICOM:** not yet supported; use PNG/JPEG for now. DICOM + window/level is planned.
- **Single-slice inference:** 2.5D/3D support is on the roadmap.
- **GAN stability:** if training diverges, try:
  - Lower LR to `1e-4`
  - Reduce `loss.l1` or `loss.ssim`
  - Smaller batch size (e.g., `4`)
- **Colab path errors:** set `DATA_DIR` explicitly in your notebook.

---

## Reproducibility Checklist
- Python **3.12**; PyTorch **≥ 2.1** (Colab defaults OK)
- Colab GPU enabled (**Runtime → Change runtime type → GPU**)
- Dataset at `/content/drive/MyDrive/training` with **4** subfolders
- Train with `src/config/train_denoise.yaml` or `src/config/train_inpaint.yaml`
- Keep seeds/configs in `src/config/*.yaml` under version control

---

## Citations
- Goodfellow et al., *Generative Adversarial Nets*, NeurIPS 2014  
- Isola et al., *Image-to-Image Translation with Conditional Adversarial Networks*, CVPR 2017  
- Ronneberger et al., *U-Net*, MICCAI 2015  
- Wang et al., *SSIM*, IEEE T-IP 2004  
- *(Add your dataset’s official citation)*

---

## Author
**Aslesha Sanjana Kodavali**  
Email: **sanjanakodavali10@gmail.com**  
LinkedIn: https://www.linkedin.com/in/sanjana-kodavali-458555245  
GitHub: https://github.com/sanjanakodavali

