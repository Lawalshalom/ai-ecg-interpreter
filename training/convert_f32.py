"""
One-time conversion: float64 .npy waveforms -> float32 .npy
Halves file size (17GB -> 8.7GB for train) and removes per-sample
dtype conversion overhead during training.

Run once:
    python training/convert_f32.py
"""
import os, sys, time
import numpy as np

DATA_DIR = "/Users/mac/Downloads/echonext-a-dataset-for-detecting-echocardiogram-confirmed-structural-heart-disease-from-ecgs-1.1.0"
OUT_DIR  = "/Users/mac/Downloads/ecg-interpreter/data"
os.makedirs(OUT_DIR, exist_ok=True)

FILES = [
    ("EchoNext_train_waveforms.npy",   "train_waveforms_f32.npy"),
    ("EchoNext_val_waveforms.npy",     "val_waveforms_f32.npy"),
    ("EchoNext_test_waveforms.npy",    "test_waveforms_f32.npy"),
]

for src_name, dst_name in FILES:
    src = os.path.join(DATA_DIR, src_name)
    dst = os.path.join(OUT_DIR, dst_name)
    if os.path.exists(dst):
        print(f"  Already exists, skipping: {dst_name}")
        continue
    print(f"Converting {src_name} -> {dst_name} ...", flush=True)
    t0 = time.time()
    arr = np.load(src, mmap_mode="r")
    print(f"  Shape: {arr.shape}  dtype: {arr.dtype}  size: {arr.nbytes/1e9:.1f} GB", flush=True)
    # Convert in chunks to avoid holding the whole array in RAM at once
    chunk = 1000
    f32 = np.lib.format.open_memmap(dst, mode="w+", dtype=np.float32, shape=arr.shape)
    for start in range(0, len(arr), chunk):
        end = min(start + chunk, len(arr))
        f32[start:end] = arr[start:end].astype(np.float32)
        pct = end / len(arr) * 100
        print(f"  {pct:5.1f}%  ({end}/{len(arr)})", end="\r", flush=True)
    del f32
    elapsed = time.time() - t0
    dst_size = os.path.getsize(dst) / 1e9
    print(f"\n  Done in {elapsed/60:.1f} min -> {dst_size:.1f} GB saved at {dst}")

print("\nAll conversions complete.")
