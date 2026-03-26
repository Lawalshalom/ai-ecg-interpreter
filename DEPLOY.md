# Deploying to Streamlit Community Cloud

## Prerequisites

- GitHub account with this repo pushed (public or private)
- [Streamlit Community Cloud](https://streamlit.io/cloud) account (free)

---

## Step 1 — Push the repo to GitHub

```bash
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/<your-username>/ecg-interpreter.git
git push -u origin main
```

> **Note:** Model weights (`checkpoints/*.pt`) and datasets (`data/`) are excluded by `.gitignore`
> because they are too large for GitHub. The app runs in demo mode without them
> (model status will show a warning in the sidebar). See [Model Weights](#model-weights) below.

---

## Step 2 — Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in.
2. Click **New app**.
3. Select your repository and branch (`main`).
4. Set **Main file path** to: `app/app.py`
5. Click **Deploy**.

---

## How dependencies are installed

Streamlit Cloud reads two files automatically:

| File | Purpose |
|------|---------|
| `requirements.txt` | Python packages (pip) |
| `packages.txt` | System (apt) packages |

`packages.txt` in this repo installs the Linux system libraries that `opencv` and `pdf2image` require:

```
libglib2.0-0    ← OpenCV core dependency
libsm6          ← OpenCV display (headless mode)
libxext6        ← OpenCV display (headless mode)
libxrender1     ← OpenCV display (headless mode)
poppler-utils   ← pdf2image PDF rendering
```

`requirements.txt` uses `opencv-python-headless` (not `opencv-python`) — the headless variant
has no GUI dependencies and works correctly in a server environment.

---

## Model Weights

The trained `.pt` model files are not included in the repo. Without them the app still launches —
the sidebar will display:

> _EchoNext model not trained. Run `python training/train.py`_

To deploy with a working model you have two options:

### Option A — Streamlit secrets + cloud storage (recommended)

Upload your `.pt` files to an S3 bucket or Google Cloud Storage, then load them at startup
using a URL stored in `.streamlit/secrets.toml`:

```toml
# .streamlit/secrets.toml  (never commit this file)
MODEL_URL = "https://your-bucket.s3.amazonaws.com/best_model.pt"
```

### Option B — Include small models in git via Git LFS

If your model files are under ~100 MB:

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes checkpoints/best_model.pt
git commit -m "Add model weights via LFS"
git push
```

---

## Troubleshooting

| Error | Fix |
|-------|-----|
| `ImportError: libGL.so.1` | Ensure `libglib2.0-0` is in `packages.txt` and you are using `opencv-python-headless` |
| `pdf2image.exceptions.PDFInfoNotInstalledError` | Ensure `poppler-utils` is in `packages.txt` |
| `No module named 'cv2'` | Confirm `opencv-python-headless>=4.8.0` is in `requirements.txt` |
| App crashes on model load | Model weights missing — see [Model Weights](#model-weights) above |
| Memory limit exceeded | PyTorch is large (~800 MB); Streamlit Cloud free tier allows 1 GB. Use CPU-only torch if needed: change `torch>=2.0.0` in `requirements.txt` to `torch>=2.0.0+cpu --find-links https://download.pytorch.org/whl/torch_stable.html` |

---

## Local run

```bash
pip install -r requirements.txt
streamlit run app/app.py
```
