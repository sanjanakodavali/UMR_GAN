# UMR-GAN: Image Restoration / Translation on 4-Class MRI Dataset

This project sets up a **GAN-based image restoration / image-to-image translation** workflow on a brain MRI dataset organized into **four class folders** (e.g., `glioma/`, `meningioma/`, `pituitary/`, `notumor/`). The repository includes a Colab-ready setup notebook to verify your environment, load the dataset from Google Drive, and run early EDA (counts, sample grids, image size distributions) plus a PyTorch `DataLoader` smoke test.

---

## Quick Start (Colab)

1. **Upload your dataset to Google Drive** at:  
   `My Drive/training/`  
   (It must contain exactly 4 subfolders, one per class, each with images.)
2. Open **Colab** and run the notebook:
   - `setup_drive_only.ipynb` (recommended) — mounts Drive and **uses your Drive folder directly** (no downloads), or
   - `setup_colab.ipynb` — multiple ingest options (Drive folder, Drive link via `gdown`, or local ZIP upload), or
   - `setup.ipynb` — simple Drive path variant.
3. In the notebook, if needed, set:
   ```python
   from pathlib import Path
   DATA_DIR = Path('/content/drive/MyDrive/training')
   ```
4. **Run all cells**. You should see:
   - Environment & GPU check
   - Class counts table (4 classes)
   - Corrupt-file scan summary
   - Random image grid
   - Image size histograms + scatter
   - PyTorch `ImageFolder` + `DataLoader` sample batch

> If your folder is not exactly `MyDrive/training`, update the path accordingly.

---

## Local Installation (Optional)

> Colab is recommended to avoid CUDA setup. If you prefer local runs:

```bash
# Clone and enter
git clone <your-repo-url>.git
cd <your-repo-name>

# Create environment (Python 3.12 recommended)
conda create -n umr-gan python=3.12 -y
conda activate umr-gan

# Install packages (CPU)
pip install torch torchvision pillow matplotlib pandas jupyter nbformat gdown

# (Optional) If you have a CUDA GPU, install the matching torch build from https://pytorch.org/get-started/locally/
# Example: pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# Launch Jupyter
jupyter lab
```

Then open `setup.ipynb` and set your local `DATA_DIR` to your dataset folder (same 4-class structure).

---

## How to Run the Notebook

- **Colab:** Upload `setup_drive_only.ipynb` to Colab and **Run all**.  
  It will mount Drive and look for `/content/drive/MyDrive/training`.  
  If your path differs, set `DATA_DIR` manually in the notebook.
- **Local:** Open `setup.ipynb` with Jupyter/Lab, set `DATA_DIR` to point to your dataset, then run all.

The notebook will **save visible outputs** (plots and tables) and print a final success message when all checks pass.

---

## Dataset Information

- **Expected layout in Drive**
  ```
  My Drive/
    training/
      class_a/
      class_b/
      class_c/
      class_d/
  ```
  Replace `class_*` with your actual class names (e.g., `glioma`, `meningioma`, `pituitary`, `notumor`).

- **Format:** Images (`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.gif`) in four subfolders, one per class.
- **Source:** If you are using public MRI data, cite the source accordingly.
  - Example (placeholder): BrainWeb simulated brain database — https://brainweb.bic.mni.mcgill.ca/brainweb/  
    *(Replace with your true data source and citation if different.)*
- **Ethics & licensing:** Ensure you have the right to use and share the dataset. Remove any PHI/PII and follow license terms for redistribution.

---

##  Reproducibility Checklist

- Python `3.12.x`  
- PyTorch `2.8.0` / TorchVision `0.23.0` (Colab defaults are fine)  
- Colab GPU (e.g., **Tesla T4**) enabled via **Runtime → Change runtime type → GPU**  
- Dataset at `/content/drive/MyDrive/training` with **exactly 4** image subfolders  
- Run `setup_drive_only.ipynb` top-to-bottom without errors

---

##  Repo Structure 

```
.
├── notebooks/
│   └── setup.ipynb              # Simple Drive path variant
├── README.md
└── ( GAN training code and configs)
```

> You can keep notebooks in a `notebooks/` folder or root—just update paths if you move them.

---

##  Author

**Aslesha Sanjana Kodavali**  
Email: <sanjanakodavali10@gmail.com>  
LinkedIn: https://www.linkedin.com/in/sanjana-kodavali-458555245  
GitHub: https://github.com/sanjanakodavali

---

## Acknowledgements

- PyTorch & TorchVision teams  
- Google Colab team  
- Brain MRI dataset contributors (cite your exact source)  
- Any prior repos or papers your training code is based on (e.g., Pix2Pix/conditional GANs)
