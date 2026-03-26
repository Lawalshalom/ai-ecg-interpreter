"""
ECG Digitization — Multi-Method Comparison
==========================================
Runs 5 extraction algorithms against the same ECG image, feeds every resulting
waveform through the trained EchoNextModel, and saves a side-by-side trace
comparison PDF + a plain-text results summary.

Methods
-------
A · Baseline (Color + Column Centroid)   — current production digitizer
B · Viterbi DP Path Tracing              — optimal path via dynamic programming
C · Morphological Skeleton               — skimage thinning → 1-px centerline
D · Grid Subtraction + Centroid          — morph-open to remove background grid
E · Sauvola Adaptive Threshold           — local adaptive thresh handles uneven lighting

Usage
-----
    python digitizer/multi_method_comparison.py            [uses /Users/mac/Downloads/norm.png]
    python digitizer/multi_method_comparison.py path/to/ecg.png
"""

from __future__ import annotations
import os, sys, time, warnings, math
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import scipy.signal as sp_signal
from scipy.ndimage import uniform_filter1d, label as nd_label
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.backends.backend_pdf import PdfPages

warnings.filterwarnings("ignore")

# ── Project root on path ─────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from digitizer.pipeline import ECGDigitizer          # layout detection re-used
from app.inference import ECGInferenceEngine

TARGET_SAMPLES = 2500
SAMPLES_PER_LEAD = 625

LEAD_DISPLAY = ["I", "II", "III", "aVR", "aVL", "aVF",
                "V1", "V2", "V3", "V4", "V5", "V6"]

# ── Shared: layout detection (run once per image) ────────────────────────────

def detect_layout(img_rgb: np.ndarray):
    """Return (trace_mask, row_bounds, col_bounds) via the production pipeline."""
    d = ECGDigitizer()
    meta = {"success": False, "notes": []}
    img_rgb = d._preprocess_image(img_rgb, meta)
    trace_mask = d._isolate_trace(img_rgb, meta)
    row_bounds, col_bounds = d._detect_layout(trace_mask, img_rgb, meta)
    return img_rgb, trace_mask, row_bounds, col_bounds


def crop_lead(img_rgb, trace_mask, row_bounds, col_bounds, row_idx, col_idx):
    """Return (rgb_crop, binary_crop) for one lead cell with a small inset."""
    y0, y1 = row_bounds[row_idx], row_bounds[row_idx + 1]
    x0, x1 = col_bounds[col_idx], col_bounds[col_idx + 1]
    pad_x = max(1, (x1 - x0) // 20)
    pad_y = max(1, (y1 - y0) // 10)
    rgb  = img_rgb   [y0 + pad_y : y1 - pad_y, x0 + pad_x : x1 - pad_x]
    mask = trace_mask[y0 + pad_y : y1 - pad_y, x0 + pad_x : x1 - pad_x]
    return rgb, mask


LAYOUT_3X4 = [
    (0, 0,  0), (0, 1,  3), (0, 2,  6), (0, 3,  9),
    (1, 0,  1), (1, 1,  4), (1, 2,  7), (1, 3, 10),
    (2, 0,  2), (2, 1,  5), (2, 2,  8), (2, 3, 11),
]


# ════════════════════════════════════════════════════════════════════════════
#  METHOD A — Baseline: colour isolation + column centroid
# ════════════════════════════════════════════════════════════════════════════

def _remove_baseline_wander(wave: np.ndarray) -> np.ndarray:
    if len(wave) < 20:
        return wave
    try:
        sos = sp_signal.butter(2, 0.02, btype="high", fs=1.0, output="sos")
        return sp_signal.sosfiltfilt(sos, wave.astype(float)).astype(np.float32)
    except Exception:
        bl = uniform_filter1d(wave.astype(float), size=max(5, len(wave) // 5))
        return (wave - bl).astype(np.float32)


def _centroid_from_mask(binary_crop: np.ndarray) -> np.ndarray:
    """Column-by-column centroid of foreground pixels.  Returns raw y-array."""
    h, w = binary_crop.shape
    wave = np.full(w, h / 2.0, dtype=np.float64)
    for col in range(w):
        ys = np.where(binary_crop[:, col] > 0)[0]
        if len(ys):
            wave[col] = float(ys.mean())
    return h - wave          # invert: image-y → voltage-y


def method_A(rgb_crop: np.ndarray, binary_crop: np.ndarray) -> np.ndarray:
    """Baseline: pre-computed binary mask → column centroid."""
    wave = _centroid_from_mask(binary_crop)
    w = len(wave)
    if w > 20:
        wave = uniform_filter1d(wave, size=max(2, w // 50))
    wave = _remove_baseline_wander(wave)
    return sp_signal.resample(wave, SAMPLES_PER_LEAD).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  METHOD B — Viterbi dynamic-programming path tracing
#  Adapted from Tereshchenkolab/ecg-digitize (MIT licence)
# ════════════════════════════════════════════════════════════════════════════

def _contiguous_centers(col_pixels: np.ndarray):
    """Centers of contiguous non-zero runs in a 1-D column."""
    centers = []
    start = None
    for i, v in enumerate(col_pixels):
        if v > 0 and start is None:
            start = i
        elif v == 0 and start is not None:
            centers.append((start + i - 1) / 2.0)
            start = None
    if start is not None:
        centers.append((start + len(col_pixels) - 1) / 2.0)
    return centers


def _dist(ax, ay, bx, by):
    return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)


def _angle(ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    if dx == 0 and dy == 0:
        return 0.0
    return math.atan2(dy, dx)


def _angle_sim(a1, a2):
    """Similarity in [0, 1] between two angles (1 = identical)."""
    diff = abs(a1 - a2)
    if diff > math.pi:
        diff = 2 * math.pi - diff
    return 1.0 - diff / math.pi


def method_B(rgb_crop: np.ndarray, binary_crop: np.ndarray) -> np.ndarray:
    """Viterbi DP: find the minimum-cost continuous path through candidate points."""
    h, w = binary_crop.shape
    if w == 0 or h == 0:
        return np.zeros(SAMPLES_PER_LEAD, dtype=np.float32)

    # Candidate points per column  {col: [(y, ...)]}
    cols = [_contiguous_centers(binary_crop[:, x]) for x in range(w)]

    DIST_W = 0.5
    LOOK_BACK = 3   # search up to 3 columns back

    # dp[x][yi] = (total_cost, prev_x, prev_yi, angle_to_here)
    dp = [dict() for _ in range(w)]

    # initialise first column
    for y in cols[0]:
        dp[0][y] = (0.0, None, None, 0.0)

    for x in range(1, w):
        for y in cols[x]:
            best_cost = float("inf")
            best_prev = (None, None, 0.0)
            # look back up to LOOK_BACK columns
            for look in range(1, LOOK_BACK + 1):
                px = x - look
                if px < 0:
                    break
                for py, (prev_cost, _, _, prev_angle) in dp[px].items():
                    cur_angle = _angle(px, py, x, y)
                    ang_term  = 1.0 - _angle_sim(cur_angle, prev_angle)
                    dist_term = _dist(px, py, x, y)
                    cost = prev_cost + dist_term * DIST_W + ang_term * (1 - DIST_W)
                    if cost < best_cost:
                        best_cost = cost
                        best_prev = (px, py, cur_angle)
            if best_prev[0] is None and dp[x - 1]:
                # no candidates found — connect to closest in previous column
                py_near = min(dp[x - 1].keys(), key=lambda py: abs(py - y))
                prev_cost = dp[x - 1][py_near][0]
                cur_angle = _angle(x - 1, py_near, x, y)
                best_cost = prev_cost + _dist(x - 1, py_near, x, y)
                best_prev = (x - 1, py_near, cur_angle)
            dp[x][y] = (best_cost, best_prev[0], best_prev[1], best_prev[2])

    # find best end point (last 20 columns)
    TAIL = min(20, w)
    best_end_cost = float("inf")
    end_x, end_y = w - 1, None
    for tx in range(w - TAIL, w):
        for ty, (cost, *_) in dp[tx].items():
            if cost < best_end_cost:
                best_end_cost = cost
                end_x, end_y = tx, ty

    if end_y is None:
        return method_A(rgb_crop, binary_crop)   # fallback

    # backtrack
    path = {}
    cx, cy = end_x, end_y
    while cx is not None:
        path[cx] = cy
        _, px, py, _ = dp[cx].get(cy, (0, None, None, 0))
        cx, cy = px, py

    # build signal array with linear interpolation for gaps
    signal = np.full(w, np.nan)
    for x, y in path.items():
        signal[x] = h - y   # invert

    # fill NaNs via interpolation
    xs = np.where(~np.isnan(signal))[0]
    if len(xs) > 1:
        signal = np.interp(np.arange(w), xs, signal[xs])
    elif len(xs) == 1:
        signal[:] = signal[xs[0]]

    if w > 20:
        signal = uniform_filter1d(signal, size=max(2, w // 50))
    signal = _remove_baseline_wander(signal.astype(np.float32))
    return sp_signal.resample(signal, SAMPLES_PER_LEAD).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  METHOD C — Morphological skeleton (1-px thinning) + centroid
# ════════════════════════════════════════════════════════════════════════════

def method_C(rgb_crop: np.ndarray, binary_crop: np.ndarray) -> np.ndarray:
    """Skeletonize the binary trace to a 1-pixel-wide line, then column centroid."""
    from skimage.morphology import skeletonize

    # Ensure proper binary bool array for skimage
    bw = (binary_crop > 0).astype(bool)

    # Widen trace slightly before thinning to bridge tiny gaps
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    bw_dilated = cv2.dilate(bw.astype(np.uint8), kernel, iterations=1).astype(bool)

    skel = skeletonize(bw_dilated).astype(np.uint8) * 255

    wave = _centroid_from_mask(skel)
    h = binary_crop.shape[0]
    w = len(wave)

    if w > 20:
        wave = uniform_filter1d(wave, size=max(2, w // 50))
    wave = _remove_baseline_wander(wave.astype(np.float32))
    return sp_signal.resample(wave, SAMPLES_PER_LEAD).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  METHOD D — Grid subtraction via morphological opening + centroid
# ════════════════════════════════════════════════════════════════════════════

def method_D(rgb_crop: np.ndarray, binary_crop: np.ndarray) -> np.ndarray:
    """
    Remove the ECG grid via morphological opening on the *grayscale* image,
    leaving only the darker trace signal.  Then column centroid.

    Logic:
      - Convert to grayscale
      - Opening with a wide horizontal kernel approximates the local background
        (grid lines are periodic thin features → survive opening; thick regions removed)
      - Actually we want to REMOVE the grid:  grid = opening(inverted_gray, h-kernel)
        Subtract grid from inverted image → trace only
    """
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY).astype(np.float32)

    # Invert so trace (dark) → bright, background → dark
    inv = 255.0 - gray

    # Horizontal opening: keeps horizontal structures ≥ kernel width (grid lines)
    # and removes narrow vertical traces — so this models the GRID background
    h_k = max(5, rgb_crop.shape[1] // 30)
    h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_k, 1))
    grid_bg = cv2.morphologyEx(inv.astype(np.uint8), cv2.MORPH_OPEN, h_kernel).astype(np.float32)

    # Vertical opening: removes horizontal grid, keeps vertical/diagonal signal
    v_k = max(3, rgb_crop.shape[0] // 15)
    v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_k))
    grid_bg_v = cv2.morphologyEx(inv.astype(np.uint8), cv2.MORPH_OPEN, v_kernel).astype(np.float32)

    # Combined background: min of both openings (keeps anything horizontal OR vertical)
    grid_combined = np.minimum(grid_bg, grid_bg_v)

    # Subtract background → isolate trace signal
    diff = np.clip(inv - grid_combined, 0, 255).astype(np.uint8)

    # Threshold the difference
    _, binary = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphological cleanup
    kernel_clean = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_clean)

    if binary.sum() == 0:   # fallback if grid removal wiped everything
        return method_A(rgb_crop, binary_crop)

    wave = _centroid_from_mask(binary)
    w = len(wave)
    if w > 20:
        wave = uniform_filter1d(wave, size=max(2, w // 50))
    wave = _remove_baseline_wander(wave.astype(np.float32))
    return sp_signal.resample(wave, SAMPLES_PER_LEAD).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  METHOD E — Sauvola local adaptive threshold + column centroid
# ════════════════════════════════════════════════════════════════════════════

def method_E(rgb_crop: np.ndarray, binary_crop: np.ndarray) -> np.ndarray:
    """
    Sauvola adaptive thresholding: computes a local threshold at each pixel
    based on local mean and standard deviation.  Better than global Otsu for
    images with uneven illumination (phone photos, photocopies).
    """
    from skimage.filters import threshold_sauvola

    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY)

    # Sauvola window ~5% of image width, at least 15 px
    win = max(15, (gray.shape[1] // 20) | 1)   # must be odd
    if win % 2 == 0:
        win += 1

    thresh = threshold_sauvola(gray, window_size=win, k=0.2)
    binary = (gray < thresh).astype(np.uint8) * 255

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if binary.sum() == 0:
        return method_A(rgb_crop, binary_crop)

    wave = _centroid_from_mask(binary)
    w = len(wave)
    if w > 20:
        wave = uniform_filter1d(wave, size=max(2, w // 50))
    wave = _remove_baseline_wander(wave.astype(np.float32))
    return sp_signal.resample(wave, SAMPLES_PER_LEAD).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  METHOD F — FFT grid-frequency suppression + centroid
# ════════════════════════════════════════════════════════════════════════════

def method_F(rgb_crop: np.ndarray, binary_crop: np.ndarray) -> np.ndarray:
    """
    Suppress the ECG grid by zeroing its dominant spatial frequencies in the
    2-D FFT of the grayscale image, then threshold and column centroid.

    The grid generates strong periodic peaks in frequency space; zeroing a
    band around those peaks removes the grid while preserving the broadband
    ECG trace signal.
    """
    gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
    inv  = 255.0 - gray   # dark trace → high values

    # 2D FFT
    F    = np.fft.fft2(inv)
    Fsh  = np.fft.fftshift(F)
    mag  = np.abs(Fsh)

    h, w = gray.shape
    cy, cx = h // 2, w // 2

    # Find dominant periodic frequencies (peaks in the power spectrum > 5× median)
    # but exclude DC (centre ± 5 px)
    mag_no_dc = mag.copy()
    mag_no_dc[cy - 5 : cy + 6, cx - 5 : cx + 6] = 0
    threshold = np.percentile(mag_no_dc, 99.5)

    # Build suppression mask: zero out strong peaks and their harmonics
    mask = np.ones_like(Fsh)
    peaks = np.argwhere(mag_no_dc > threshold)
    for py, px in peaks:
        # zero a small neighbourhood around each peak
        r = max(2, min(h, w) // 80)
        y0 = max(0, py - r); y1 = min(h, py + r + 1)
        x0 = max(0, px - r); x1 = min(w, px + r + 1)
        mask[y0:y1, x0:x1] = 0

    # Apply mask and inverse FFT
    Fsh_filtered = Fsh * mask
    F_filtered   = np.fft.ifftshift(Fsh_filtered)
    result       = np.real(np.fft.ifft2(F_filtered))

    # Normalise and threshold
    result = np.clip(result, 0, None)
    result = (result / (result.max() + 1e-6) * 255).astype(np.uint8)
    _, binary = cv2.threshold(result, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

    if binary.sum() == 0:
        return method_A(rgb_crop, binary_crop)

    wave = _centroid_from_mask(binary)
    wl = len(wave)
    if wl > 20:
        wave = uniform_filter1d(wave, size=max(2, wl // 50))
    wave = _remove_baseline_wander(wave.astype(np.float32))
    return sp_signal.resample(wave, SAMPLES_PER_LEAD).astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  Post-processing: tile, normalise
# ════════════════════════════════════════════════════════════════════════════

def build_waveform(leads: list) -> np.ndarray:
    """Tile each 625-sample lead × 4 → (2500, 12) float32."""
    wf = np.zeros((TARGET_SAMPLES, 12), dtype=np.float32)
    for i, lead in enumerate(leads):
        if lead is not None and len(lead):
            wf[:, i] = np.tile(lead, 4)[:TARGET_SAMPLES]
    return wf


def normalize_echonext(wf: np.ndarray) -> np.ndarray:
    out = wf.copy()
    for i in range(out.shape[1]):
        s = out[:, i].std()
        if s > 1e-6:
            out[:, i] = np.clip((out[:, i] - out[:, i].mean()) / s, -5.0, 5.0)
    return out.astype(np.float32)


# ════════════════════════════════════════════════════════════════════════════
#  Main comparison runner
# ════════════════════════════════════════════════════════════════════════════

METHODS = {
    "A · Baseline\n(Color+Centroid)":    method_A,
    "B · Viterbi\n(DP Path)":            method_B,
    "C · Skeleton\n(Morph Thin)":        method_C,
    "D · Grid Subtract\n(Morph Open)":   method_D,
    "E · Sauvola\n(Adaptive Thresh)":    method_E,
    "F · FFT Grid\n(Freq Suppress)":     method_F,
}

METHOD_COLORS = {
    "A · Baseline\n(Color+Centroid)":    "#2563eb",   # blue
    "B · Viterbi\n(DP Path)":            "#16a34a",   # green
    "C · Skeleton\n(Morph Thin)":        "#9333ea",   # purple
    "D · Grid Subtract\n(Morph Open)":   "#ea580c",   # orange
    "E · Sauvola\n(Adaptive Thresh)":    "#dc2626",   # red
    "F · FFT Grid\n(Freq Suppress)":     "#0891b2",   # cyan
}


def run_all_methods(image_path: str) -> dict:
    """
    Process image with all methods.
    Returns dict: method_name → {"waveform": ndarray, "leads": list, "time_s": float}
    """
    img_bgr = cv2.imread(image_path)
    if img_bgr is None:
        raise FileNotFoundError(f"Cannot load: {image_path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print(f"Image loaded: {img_rgb.shape[1]}×{img_rgb.shape[0]} px")
    print("Detecting layout...")
    img_rgb, trace_mask, row_bounds, col_bounds = detect_layout(img_rgb)
    print(f"  Row bounds: {row_bounds}")
    print(f"  Col bounds: {col_bounds}")

    n_rows = len(row_bounds) - 1
    n_cols = len(col_bounds) - 1

    results = {}
    for method_name, extract_fn in METHODS.items():
        short = method_name.split("\n")[0]
        print(f"\nRunning {short} ...", end="", flush=True)
        t0 = time.time()
        leads = [None] * 12
        errors = 0

        for row_idx, col_idx, lead_idx in LAYOUT_3X4:
            if row_idx >= n_rows or col_idx >= n_cols:
                errors += 1
                continue
            try:
                rgb_crop, bin_crop = crop_lead(
                    img_rgb, trace_mask, row_bounds, col_bounds, row_idx, col_idx
                )
                leads[lead_idx] = extract_fn(rgb_crop, bin_crop)
            except Exception as e:
                print(f" [lead {lead_idx} err: {e}]", end="", flush=True)
                errors += 1

        # fill missing with zeros
        for i in range(12):
            if leads[i] is None:
                leads[i] = np.zeros(SAMPLES_PER_LEAD, dtype=np.float32)

        wf = normalize_echonext(build_waveform(leads))
        elapsed = time.time() - t0
        results[method_name] = {
            "waveform": wf,
            "leads": leads,
            "time_s": elapsed,
            "errors": errors,
        }
        print(f" done in {elapsed:.1f}s  (errors={errors})")

    return results


# ════════════════════════════════════════════════════════════════════════════
#  Plotting
# ════════════════════════════════════════════════════════════════════════════

def _signal_quality(wave: np.ndarray) -> tuple:
    """Return (snr_db, smoothness) quality metrics for a single lead."""
    # SNR: ratio of signal power to high-freq noise power
    if wave.std() < 1e-6:
        return 0.0, 0.0
    # Noise ≈ high-passed version (>50 Hz equivalent)
    sos = sp_signal.butter(4, 0.4, btype="high", fs=1.0, output="sos")
    noise = sp_signal.sosfiltfilt(sos, wave)
    snr = 10 * np.log10((wave.var() + 1e-12) / (noise.var() + 1e-12))
    # Smoothness: inverse of mean absolute second derivative (lower = smoother)
    d2 = np.diff(wave, n=2)
    smoothness = 1.0 / (np.mean(np.abs(d2)) + 1e-6)
    return float(snr), float(smoothness)


def save_comparison_pdf(results: dict, inference_results: dict,
                        image_path: str, output_path: str):
    """
    Save a multi-page PDF:
      Page 1  — 12-lead × 6-method waveform grid
      Page 2  — Per-method model predictions bar chart
      Page 3  — Signal quality metrics (SNR, smoothness) heatmap
      Page 4  — Summary table
    """
    methods = list(results.keys())
    n_methods = len(methods)
    t = np.linspace(0, 2.5, SAMPLES_PER_LEAD)   # 2.5 s (first tile)

    with PdfPages(output_path) as pdf:

        # ── PAGE 1: Waveform grid ──────────────────────────────────────────
        fig = plt.figure(figsize=(4 * n_methods, 2.8 * 12), constrained_layout=False)
        fig.suptitle(
            f"ECG Digitisation — Method Comparison\n{Path(image_path).name}",
            fontsize=16, fontweight="bold", y=0.995,
        )
        gs = gridspec.GridSpec(12, n_methods, figure=fig,
                               hspace=0.08, wspace=0.05,
                               top=0.985, bottom=0.01, left=0.06, right=0.99)

        for lead_i, lead_name in enumerate(LEAD_DISPLAY):
            for m_i, (method_name, res) in enumerate(results.items()):
                ax = fig.add_subplot(gs[lead_i, m_i])
                wave_full = res["waveform"][:, lead_i]
                # Show first tile only (2.5 s)
                wave_show = wave_full[:SAMPLES_PER_LEAD]
                color = METHOD_COLORS[method_name]
                ax.plot(t, wave_show, color=color, linewidth=0.7, rasterized=True)
                ax.set_ylim(-4.5, 4.5)
                ax.axhline(0, color="gray", linewidth=0.3, alpha=0.5)
                ax.tick_params(left=False, bottom=False,
                               labelleft=False, labelbottom=False)
                ax.spines[["top", "right", "bottom", "left"]].set_visible(False)

                if lead_i == 0:
                    short = method_name.replace("\n", " ")
                    ax.set_title(short, fontsize=7.5, fontweight="bold",
                                 color=color, pad=3)
                if m_i == 0:
                    ax.set_ylabel(lead_name, rotation=0, labelpad=28,
                                  fontsize=9, fontweight="bold", va="center")

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── PAGE 2: Model prediction bars ─────────────────────────────────
        conditions = inference_results[methods[0]]["labels"]
        n_cond = len(conditions)
        fig, axes = plt.subplots(n_cond, 1,
                                 figsize=(14, n_cond * 1.1 + 2),
                                 constrained_layout=True)
        fig.suptitle("Model Predictions per Method (SHD probabilities)",
                     fontsize=13, fontweight="bold")

        bar_width = 0.8 / n_methods
        x = np.arange(n_cond)

        for m_i, method_name in enumerate(methods):
            probs = inference_results[method_name]["probabilities"]
            color = METHOD_COLORS[method_name]
            short = method_name.replace("\n", " ")
            for ax_i, ax in enumerate(axes):
                offset = (m_i - n_methods / 2 + 0.5) * bar_width
                ax.bar(ax_i + offset, probs[ax_i],
                       width=bar_width, color=color,
                       label=short if ax_i == 0 else None,
                       alpha=0.85, edgecolor="white", linewidth=0.5)

        for ax_i, ax in enumerate(axes):
            ax.set_xlim(ax_i - 0.5, ax_i + 0.5)
            cond_short = conditions[ax_i].replace(" (Mod+)", "").replace(" (Composite)", " ★")
            ax.set_ylabel(cond_short, rotation=0, labelpad=5,
                          fontsize=7.5, ha="right", va="center")
            ax.axhline(0.60, color="red",   linewidth=0.8,
                       linestyle="--", alpha=0.6)
            ax.axhline(0.35, color="orange", linewidth=0.8,
                       linestyle="--", alpha=0.5)
            ax.set_ylim(0, 1.05)
            ax.yaxis.set_tick_params(labelsize=6)
            ax.set_xticks([])
            ax.spines[["top", "right", "bottom"]].set_visible(False)

        # Single legend
        handles = [
            plt.Rectangle((0, 0), 1, 1,
                           color=METHOD_COLORS[m],
                           label=m.replace("\n", " "))
            for m in methods
        ]
        fig.legend(handles=handles, loc="lower center",
                   ncol=n_methods, fontsize=8.5,
                   bbox_to_anchor=(0.5, -0.01), framealpha=0.9)
        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── PAGE 3: Signal quality heatmap ────────────────────────────────
        snr_matrix  = np.zeros((12, n_methods))
        smo_matrix  = np.zeros((12, n_methods))
        for m_i, method_name in enumerate(methods):
            wf = results[method_name]["waveform"]
            for li in range(12):
                snr, smo = _signal_quality(wf[:SAMPLES_PER_LEAD, li])
                snr_matrix[li, m_i] = snr
                smo_matrix[li, m_i] = np.log1p(smo)  # log scale

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7),
                                        constrained_layout=True)
        fig.suptitle("Signal Quality Metrics per Lead × Method",
                     fontsize=13, fontweight="bold")

        method_labels = [m.replace("\n", " ") for m in methods]

        im1 = ax1.imshow(snr_matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=20)
        ax1.set_xticks(range(n_methods))
        ax1.set_xticklabels(method_labels, rotation=30, ha="right", fontsize=8)
        ax1.set_yticks(range(12))
        ax1.set_yticklabels(LEAD_DISPLAY, fontsize=9)
        ax1.set_title("SNR (dB)  — higher=better", fontweight="bold")
        plt.colorbar(im1, ax=ax1, shrink=0.8)
        for li in range(12):
            for mi in range(n_methods):
                ax1.text(mi, li, f"{snr_matrix[li, mi]:.0f}",
                         ha="center", va="center", fontsize=7)

        im2 = ax2.imshow(smo_matrix, aspect="auto", cmap="RdYlGn")
        ax2.set_xticks(range(n_methods))
        ax2.set_xticklabels(method_labels, rotation=30, ha="right", fontsize=8)
        ax2.set_yticks(range(12))
        ax2.set_yticklabels(LEAD_DISPLAY, fontsize=9)
        ax2.set_title("Smoothness log(1+1/|Δ²|)  — higher=better",
                      fontweight="bold")
        plt.colorbar(im2, ax=ax2, shrink=0.8)

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

        # ── PAGE 4: Summary table ─────────────────────────────────────────
        shd_probs  = [inference_results[m]["shd_risk"]  for m in methods]
        risk_levs  = [inference_results[m]["risk_level"] for m in methods]
        proc_times = [results[m]["time_s"] for m in methods]
        avg_snrs   = [snr_matrix[:, mi].mean() for mi in range(n_methods)]

        fig, ax = plt.subplots(figsize=(14, 5))
        ax.axis("off")
        fig.suptitle("Summary Table — All Methods", fontsize=14, fontweight="bold")

        col_headers = ["Method", "SHD Risk", "Risk Level",
                       "Avg SNR (dB)", "Time (s)"]
        table_data = []
        for m_i, method_name in enumerate(methods):
            row = [
                method_name.replace("\n", " "),
                f"{shd_probs[m_i]:.1%}",
                risk_levs[m_i],
                f"{avg_snrs[m_i]:.1f}",
                f"{proc_times[m_i]:.1f}",
            ]
            table_data.append(row)

        table = ax.table(
            cellText=table_data,
            colLabels=col_headers,
            cellLoc="center",
            loc="center",
        )
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.3, 2.2)

        # Colour header row
        for j in range(len(col_headers)):
            table[0, j].set_facecolor("#1e40af")
            table[0, j].set_text_props(color="white", fontweight="bold")

        # Colour risk cells
        risk_colors = {"LOW": "#bbf7d0", "MODERATE": "#fef9c3", "HIGH": "#fecaca"}
        for m_i, method_name in enumerate(methods):
            rl = risk_levs[m_i]
            table[m_i + 1, 2].set_facecolor(risk_colors.get(rl, "white"))
            table[m_i + 1, 0].set_facecolor(METHOD_COLORS[method_name] + "33")

        pdf.savefig(fig, dpi=150)
        plt.close(fig)

    print(f"\nPDF saved → {output_path}")


# ════════════════════════════════════════════════════════════════════════════
#  Text report
# ════════════════════════════════════════════════════════════════════════════

def save_text_report(results: dict, inference_results: dict,
                     image_path: str, output_path: str):
    """Write a plain-text summary to output_path."""
    lines = []
    sep = "=" * 72

    lines += [
        sep,
        "ECG DIGITISATION — MULTI-METHOD COMPARISON",
        f"Image : {image_path}",
        f"Date  : {time.strftime('%Y-%m-%d %H:%M:%S')}",
        sep, "",
    ]

    methods = list(results.keys())

    # Per-method inference results
    conditions = inference_results[methods[0]]["labels"]

    # Header
    method_shorts = [m.split("\n")[0] for m in methods]
    col_w = 14
    header = f"{'Condition':<45}" + "".join(f"{s:>{col_w}}" for s in method_shorts)
    lines.append(header)
    lines.append("-" * len(header))

    for ci, cond in enumerate(conditions):
        row = f"{cond:<45}"
        for m in methods:
            prob = inference_results[m]["probabilities"][ci]
            flag = "▲HIGH" if prob >= 0.60 else ("~MED" if prob >= 0.35 else "  low")
            row += f"  {prob:5.1%} {flag:5s}"
        lines.append(row)

    lines.append("-" * len(header))
    row = f"{'SHD Composite':<45}"
    for m in methods:
        row += f"  {inference_results[m]['shd_risk']:5.1%}      "
    lines.append(row)

    row = f"{'Risk Level':<45}"
    for m in methods:
        row += f"  {inference_results[m]['risk_level']:12s}"
    lines.append(row)

    lines += ["", sep, "SIGNAL QUALITY (avg SNR dB across 12 leads)", sep]
    for m in methods:
        wf = results[m]["waveform"]
        snrs = [_signal_quality(wf[:SAMPLES_PER_LEAD, li])[0] for li in range(12)]
        per_lead = "  ".join(f"{LEAD_DISPLAY[li]}:{snrs[li]:.0f}" for li in range(12))
        lines.append(f"{m.split(chr(10))[0]:<30}  avg={np.mean(snrs):.1f} dB")
        lines.append(f"  {per_lead}")
        lines.append(f"  Proc time: {results[m]['time_s']:.2f}s")
        lines.append("")

    lines += [sep, "ECG FEATURES (estimated from Method A baseline)", sep]
    for k, v in inference_results[methods[0]].get("ecg_features", {}).items():
        lines.append(f"  {k}: {v}")

    lines += ["", sep]

    with open(output_path, "w") as f:
        f.write("\n".join(lines))
    print(f"Text report saved → {output_path}")


# ════════════════════════════════════════════════════════════════════════════
#  Entry point
# ════════════════════════════════════════════════════════════════════════════

def main(image_path: str, output_dir: str = None):
    if output_dir is None:
        output_dir = str(ROOT / "outputs")
    os.makedirs(output_dir, exist_ok=True)

    stem = Path(image_path).stem
    pdf_path  = os.path.join(output_dir, f"{stem}_method_comparison.pdf")
    txt_path  = os.path.join(output_dir, f"{stem}_method_comparison.txt")

    print(f"\n{'='*60}")
    print("  ECG DIGITISATION — MULTI-METHOD COMPARISON")
    print(f"  Input : {image_path}")
    print(f"  Output: {output_dir}/")
    print(f"{'='*60}\n")

    # 1. Run all methods
    results = run_all_methods(image_path)

    # 2. Run inference on each waveform
    print("\nRunning model inference on all methods...")
    engine = ECGInferenceEngine()
    inference_results = {}
    for method_name, res in results.items():
        ir = engine.predict(res["waveform"], {})
        inference_results[method_name] = ir
        short = method_name.split("\n")[0]
        print(f"  {short:<30} SHD={ir['shd_risk']:.1%}  {ir['risk_level']}")

    # 3. Save outputs
    print()
    save_comparison_pdf(results, inference_results, image_path, pdf_path)
    save_text_report(results, inference_results, image_path, txt_path)

    print(f"\nAll done.  Files saved to {output_dir}/")
    return pdf_path, txt_path


if __name__ == "__main__":
    img = sys.argv[1] if len(sys.argv) > 1 else "/Users/mac/Downloads/norm.png"
    main(img)
