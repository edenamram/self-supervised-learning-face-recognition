
# SSL Face Super‑Resolution (MNTSR‑based)

Create **high‑resolution faces from low‑resolution inputs** using a self‑supervised learning (SSL) setup inspired by the MNTSR baseline. Training data is **CelebA**, and we provide links to the train/test sets below.

---

## What’s in this repo
- `SSL_Face_SR_MNTSR_Final_Version.ipynb` — full pipeline notebook (data setup, training, and inference).
- (Optional) You can export the notebook to a Python script and run it as CLI (see below).

---

## Data

**Train/Test (Google Drive):**
- **Train**: https://drive.google.com/drive/folders/1SugVuD0SaIw5FOmboAxHgFIJRJUTiKs9?usp=drive_link
- **Test**:  https://drive.google.com/drive/folders/1snFmbg0i2y8AMWXnjPEhEq4rre9AgkkH?usp=drive_link

---

## Environment

- **Python**: 3.10+
- Recommended GPU with CUDA (works on CPU but slower).

Install dependencies:
```bash
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -U pip
pip install -r requirements.txt
```

If you don’t have a `requirements.txt` yet, start with this and adjust if needed:
```
torch>=2.1
torchvision>=0.16
numpy
pillow
opencv-python
tqdm
scikit-image
matplotlib
albumentations
```

---

## Run on Google Colab (easiest)

1) Open the notebook in Colab:  
https://colab.research.google.com/github/edenamram/self-supervised-learning-face-recognition/blob/main/SSL_Face_SR_MNTSR_Final_Version.ipynb

2) In the first cell, set:
- `TRAIN_DIR` → path to your **CelebA train** folder (from Drive link above)
- `TEST_DIR`  → path to your **CelebA test** folder
- `OUTPUT_DIR` and `CHECKPOINT_DIR` (optional)

3) Run all cells. The notebook:
- Prepares data / augmentations for **SSL**.
- Trains the MNTSR‑style SR network with self‑supervised objectives.
- Saves checkpoints into `checkpoints/`.
- Runs **inference** on test images and writes results to `outputs/` (PSNR/SSIM/LPIPS optional).

---

## How we use **MNTSR** + SSL (short)
- We adopt an MNTSR‑style SR backbone and train it with **self‑supervised** losses/augmentations so it learns to reconstruct HR faces from LR inputs **without identity labels**.
- The SSL setup uses strong augmentations and a reconstruction objective; you can optionally log **PSNR/SSIM/LPIPS** and track face‑specific quality improvements.
- Replace/extend the loss functions or the backbone in the notebook as you iterate.

---

## Troubleshooting
- **CUDA not found** → install a Torch build matching your CUDA. Or run on CPU by installing the CPU‑only Torch wheel.
- **Slow training on CPU** → reduce image size, batch size, and epochs for a quick sanity run.
- **Paths wrong** → verify the first notebook cell: `TRAIN_DIR`, `TEST_DIR`, `OUTPUT_DIR`, `CHECKPOINT_DIR`.

