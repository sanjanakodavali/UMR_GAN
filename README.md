# UMR-GAN — Universal Medical Image Restoration GAN
Restore MRI slices from noise, artifacts, and missing regions using a U-Net + PatchGAN (hinge) with L1/SSIM/perceptual-like losses.

## Structure
- `data/` → raw or sample images (place datasets here)
- `notebooks/` → Jupyter notebooks (see `setup.ipynb`)
- `src/` → model and training code (medrestor_gan package)
- `ui/` → upcoming Gradio interface
- `results/` → outputs and checkpoints
- `docs/` → diagrams and project visuals

## Quickstart
```bash
pip install -r requirements.txt
jupyter notebook notebooks/setup.ipynb
```
Train (example):
```bash
python src/medrestor_gan/train.py --data data/raw --out results/checkpoints --size 256 --bs 4 --epochs 2
```
