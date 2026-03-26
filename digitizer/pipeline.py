"""
ECG Image Digitizer Pipeline — v2

Converts uploaded ECG images (JPEG, PNG, PDF, smartphone photos) into
a (2500, 12) float32 numpy array at 250Hz — matching EchoNext format.

Key improvements over v1:
  - Color-based trace isolation (black trace vs pink/red grid — no grayscale ambiguity)
  - Robust layout detection using trace-pixel projection profiles
  - Correct 2.5s lead → 10s via 4× tiling (not stretch, preserves heart rate)
  - Calibration pulse removal
  - Handles scanned paper, PDF, and smartphone photos

Standard 12-lead layout (most common clinical format):
  Row 1: I,   aVR, V1, V4   (each 2.5 s)
  Row 2: II,  aVL, V2, V5
  Row 3: III, aVF, V3, V6
  Row 4: II   (full 10 s rhythm strip — optional)

Output lead order matches EchoNext:
  0=I  1=II  2=III  3=aVR  4=aVL  5=aVF  6=V1  7=V2  8=V3  9=V4  10=V5  11=V6
"""

import io
import os
import warnings
from typing import Optional

import cv2
import numpy as np
from PIL import Image
import scipy.signal as signal
from scipy.ndimage import uniform_filter1d, label as nd_label

warnings.filterwarnings("ignore", category=UserWarning)

# Row × Col → output lead index
LAYOUT_3X4 = [
    (0, 0, 0),   # I   → 0
    (0, 1, 3),   # aVR → 3
    (0, 2, 6),   # V1  → 6
    (0, 3, 9),   # V4  → 9
    (1, 0, 1),   # II  → 1
    (1, 1, 4),   # aVL → 4
    (1, 2, 7),   # V2  → 7
    (1, 3, 10),  # V5  → 10
    (2, 0, 2),   # III → 2
    (2, 1, 5),   # aVF → 5
    (2, 2, 8),   # V3  → 8
    (2, 3, 11),  # V6  → 11
]

TARGET_SAMPLES = 2500   # 10 s × 250 Hz
SAMPLES_PER_LEAD = 625  # 2.5 s × 250 Hz (per lead in 3×4 layout)


class ECGDigitizer:
    """
    End-to-end ECG image digitizer.

    Usage:
        digitizer = ECGDigitizer()
        waveform, meta = digitizer.process("ecg.jpg")
        # waveform: np.ndarray (2500, 12) float32
    """

    def process(self, source) -> tuple[np.ndarray, dict]:
        meta = {"success": False, "notes": []}
        try:
            img_rgb = self._load_image(source, meta)
            if not self._validate_is_ecg(img_rgb, meta):
                return np.zeros((TARGET_SAMPLES, 12), dtype=np.float32), meta
            img_rgb = self._preprocess_image(img_rgb, meta)
            trace_mask = self._isolate_trace(img_rgb, meta)
            row_bounds, col_bounds = self._detect_layout(trace_mask, img_rgb, meta)
            leads = self._extract_all_leads(trace_mask, img_rgb, row_bounds, col_bounds, meta)
            waveform = self._build_waveform(leads, meta)
            waveform = self._normalize_echonext(waveform)
            meta["success"] = True
            meta["fs"] = 250
            return waveform, meta
        except Exception as e:
            import traceback
            meta["notes"].append(f"ERROR: {e}")
            meta["error"] = str(e)
            meta["traceback"] = traceback.format_exc()
            return np.zeros((TARGET_SAMPLES, 12), dtype=np.float32), meta

    # ── ECG Validation ─────────────────────────────────────────────────────────

    def _validate_is_ecg(self, img_rgb: np.ndarray, meta: dict) -> bool:
        """
        Multi-layer validation that the image is a 12-lead ECG, not a document.

        Algorithm (based on published ECG digitization literature):
          FATAL  A: resolution too small
          FATAL  B: blank image
          FATAL  C: solid-black image
          FATAL  D: portrait orientation — resumes/letters are portrait; ECGs are landscape
          Scored checks (max 8 points, need ≥ 4):
            1 (+2): HSV pink/red grid color — clinical ECG paper signature
            2 (+2): 2D FFT bidirectional spatial periodicity — grid lattice
            3 (+1): Orthogonal Hough line families (both H and V)
            4 (+1): Landscape or near-square aspect ratio
            5 (+1): Signal distributed across all 3 vertical thirds (lead rows)
            6 (+1): Long continuous horizontal traces (≥ 7% image width)
        """
        h, w = img_rgb.shape[:2]

        # ── FATAL A: minimum resolution ────────────────────────────────────────
        if h < 100 or w < 150:
            meta["error"] = (
                f"Image is too small to contain a readable ECG ({w}×{h} px). "
                "Please upload a higher-resolution scan — 800×600 px minimum recommended."
            )
            return False

        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        _, dark = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        density = float(dark.sum()) / 255.0 / (h * w)

        # ── FATAL B: blank ──────────────────────────────────────────────────────
        if density < 0.001:
            meta["error"] = (
                "The uploaded image appears blank or empty. "
                "Please upload a clear scan or photo of a 12-lead ECG printout."
            )
            return False

        # ── FATAL C: solid black ────────────────────────────────────────────────
        if density > 0.65:
            meta["error"] = (
                "The uploaded image is too dark or appears to be a solid file. "
                "Please upload a clean scan of an ECG printout."
            )
            return False

        # ── FATAL D: portrait orientation ──────────────────────────────────────
        # All standard 12-lead ECGs are landscape-oriented. Documents (resumes,
        # letters, reports) are portrait. Reject anything clearly portrait.
        aspect = w / h
        if aspect < 0.72:
            meta["error"] = (
                "The uploaded file appears to be a portrait-oriented document "
                "(e.g. a letter, resume, or report), not an ECG tracing. "
                "Standard 12-lead ECGs are printed in landscape orientation. "
                "Please upload a landscape-oriented ECG image or scan."
            )
            return False

        score = 0
        reasons_failed = []

        # ── Check 1 (+2): HSV pink/red ECG grid color ──────────────────────────
        # Clinical ECG paper has a characteristic pink/red grid.
        # OpenCV HSV: H=0-179, S/V=0-255.
        # Red/pink hue: H∈[0,15]∪[165,179], S>35, V>80
        hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]
        pink_mask = (
            ((h_ch <= 15) | (h_ch >= 165)) &
            (s_ch > 35) &
            (v_ch > 80)
        )
        pink_ratio = float(pink_mask.sum()) / (h * w)
        if pink_ratio > 0.025:
            score += 2
            meta["notes"].append(f"ECG grid color: {pink_ratio*100:.1f}% pink/red (strong)")
        elif pink_ratio > 0.006:
            score += 1
            meta["notes"].append(f"ECG grid color: {pink_ratio*100:.1f}% pink/red (faint)")
        else:
            reasons_failed.append(f"no ECG grid color ({pink_ratio*100:.2f}% pink/red)")

        # ── Check 2 (+2): 2D FFT spatial periodicity ───────────────────────────
        # ECG grid paper creates sharp periodic peaks in both the horizontal and
        # vertical frequency directions. Random documents and photos do not.
        small_h, small_w = min(256, h), min(512, w)
        small = cv2.resize(gray, (small_w, small_h),
                           interpolation=cv2.INTER_AREA).astype(float)
        f2d = np.abs(np.fft.fftshift(np.fft.fft2(small)))
        cy, cx = small_h // 2, small_w // 2
        dc_r = max(3, min(small_h, small_w) // 20)
        f2d[cy - dc_r:cy + dc_r, cx - dc_r:cx + dc_r] = 0
        h_slice = f2d[cy, :]
        v_slice = f2d[:, cx]
        h_peaks, _ = signal.find_peaks(h_slice, height=h_slice.mean() * 4, distance=4)
        v_peaks, _ = signal.find_peaks(v_slice, height=v_slice.mean() * 4, distance=4)
        if len(h_peaks) >= 3 and len(v_peaks) >= 2:
            score += 2
        elif len(h_peaks) >= 1 and len(v_peaks) >= 1:
            score += 1
        else:
            reasons_failed.append(
                f"no regular grid frequency (H-peaks:{len(h_peaks)}, V-peaks:{len(v_peaks)})"
            )

        # ── Check 3 (+1): Orthogonal Hough line families ───────────────────────
        # ECG grid has dense families of near-horizontal AND near-vertical lines.
        edges = cv2.Canny(gray, 30, 100, apertureSize=3)
        hough_thresh = max(min(w, h) // 6, 30)
        lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=hough_thresh)
        if lines is not None:
            angles = lines[:, 0, 1]
            n_horiz = int(np.sum((angles > np.radians(80)) & (angles < np.radians(100))))
            n_vert  = int(np.sum((angles < np.radians(10)) | (angles > np.radians(170))))
            if n_horiz >= 4 and n_vert >= 3:
                score += 1
                meta["notes"].append(f"Hough grid lines: {n_horiz}H {n_vert}V")
            else:
                reasons_failed.append(f"sparse orthogonal lines (H:{n_horiz}, V:{n_vert})")
        else:
            reasons_failed.append("no grid lines detected (Hough)")

        # ── Check 4 (+1): Landscape orientation ────────────────────────────────
        if w >= h:
            score += 1
        else:
            reasons_failed.append(f"near-portrait aspect ({aspect:.2f})")

        # ── Check 5 (+1): Signal in all 3 vertical thirds ──────────────────────
        # A 12-lead ECG has 3 lead rows filling all three vertical thirds.
        third = max(1, h // 3)
        band_dens = [
            float(dark[i * third:(i + 1) * third, :].sum()) / 255.0 / max(third * w, 1)
            for i in range(3)
        ]
        if sum(1 for d in band_dens if d > 0.003) >= 3:
            score += 1
        else:
            reasons_failed.append(
                f"signal absent from some lead rows "
                f"(densities: {[f'{d:.3f}' for d in band_dens]})"
            )

        # ── Check 6 (+1): Long continuous horizontal traces ────────────────────
        # ECG waveform traces span the full width of each lead column.
        # Text runs in documents are short (< 7% of image width).
        kernel_w = max(int(w * 0.07), 20)
        h_kern = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_w, 1))
        h_open = cv2.morphologyEx(dark, cv2.MORPH_OPEN, h_kern)
        n_long = int((h_open.sum(axis=1) / 255 > kernel_w).sum())
        if n_long >= max(6, h // 25):
            score += 1
        else:
            reasons_failed.append(f"too few long horizontal traces ({n_long} rows)")

        meta["notes"].append(
            f"ECG validation score: {score}/8 "
            f"(aspect={aspect:.2f}, pink={pink_ratio*100:.1f}%)"
        )

        if score < 4:
            detail = "; ".join(reasons_failed[:3]) if reasons_failed else "multiple checks failed"
            meta["error"] = (
                "The submitted file does not contain a recognizable 12-lead ECG tracing "
                f"({detail}). "
                "Please upload a JPEG, PNG, TIFF, or PDF image of a standard "
                "landscape-oriented 12-lead ECG printout."
            )
            return False

        return True

    # ── Image Loading ──────────────────────────────────────────────────────────

    def _load_image(self, source, meta: dict) -> np.ndarray:
        if isinstance(source, np.ndarray):
            if source.ndim == 2:
                return cv2.cvtColor(source, cv2.COLOR_GRAY2RGB)
            return source

        if isinstance(source, Image.Image):
            return np.array(source.convert("RGB"))

        if isinstance(source, (bytes, io.BytesIO)):
            data = source if isinstance(source, bytes) else source.read()
            if data[:4] == b"%PDF":
                return self._pdf_to_image(data, meta)
            pil = Image.open(io.BytesIO(data)).convert("RGB")
            return np.array(pil)

        path = str(source)
        if path.lower().endswith(".pdf"):
            with open(path, "rb") as f:
                return self._pdf_to_image(f.read(), meta)

        img = cv2.imread(path)
        if img is None:
            raise ValueError(f"Could not load image: {path}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    def _pdf_to_image(self, data: bytes, meta: dict) -> np.ndarray:
        try:
            from pdf2image import convert_from_bytes
            pages = convert_from_bytes(data, dpi=200, first_page=1, last_page=1)
            meta["notes"].append("PDF converted at 200 DPI")
            return np.array(pages[0].convert("RGB"))
        except ImportError:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(data))
            for img_file in reader.pages[0].images:
                pil = Image.open(io.BytesIO(img_file.data)).convert("RGB")
                return np.array(pil)
            raise ValueError("Cannot extract image from PDF — install pdf2image + poppler.")

    # ── Image Preprocessing ────────────────────────────────────────────────────

    def _preprocess_image(self, img: np.ndarray, meta: dict) -> np.ndarray:
        h, w = img.shape[:2]

        # Upscale small images for better waveform resolution
        if w < 800:
            scale = 800 / w
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            meta["notes"].append(f"Upscaled {scale:.1f}×")

        # Deskew using edge-based Hough lines
        img = self._deskew(img, meta)
        return img

    def _deskew(self, img: np.ndarray, meta: dict, max_angle: float = 8.0) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        lines = cv2.HoughLines(edges, 1, np.pi / 180,
                               threshold=max(img.shape[:2]) // 4)
        if lines is None:
            return img
        angles = []
        for rho, theta in lines[:, 0]:
            a = np.degrees(theta) - 90
            if abs(a) < max_angle:
                angles.append(a)
        if not angles:
            return img
        angle = np.median(angles)
        if abs(angle) < 0.5:
            return img
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        rotated = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC,
                                 borderMode=cv2.BORDER_REPLICATE)
        meta["notes"].append(f"Deskewed {angle:.1f}°")
        return rotated

    # ── Trace Isolation ────────────────────────────────────────────────────────

    def _isolate_trace(self, img: np.ndarray, meta: dict) -> np.ndarray:
        """
        Isolate the ECG trace using HSV + LAB color analysis.

        Color ECGs (pink/red grid): separate the black trace from the colored grid
          using LAB lightness + RGB dark-pixel thresholding.
        B&W / grayscale ECGs: CLAHE contrast normalization + Otsu × 1.2 threshold.

        The Otsu × 1.2 bias (from Tereshchenko et al. PMC9286778) preserves trace
        peak pixels that Otsu alone can discard.
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h_ch = hsv[:, :, 0]
        s_ch = hsv[:, :, 1]
        v_ch = hsv[:, :, 2]

        # Detect pink/red ECG grid (same criterion as validation)
        pink_ratio = float(
            (((h_ch <= 15) | (h_ch >= 165)) & (s_ch > 35) & (v_ch > 80)).sum()
        ) / (img.shape[0] * img.shape[1])
        is_colour = pink_ratio > 0.006

        if is_colour:
            # Color mode: isolate dark (black) ECG trace pixels.
            # Use LAB L channel (perceptually uniform lightness) + RGB backup.
            lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
            L = lab[:, :, 0]  # 0-255 in OpenCV LAB
            r = img[:, :, 0].astype(np.int16)
            g = img[:, :, 1].astype(np.int16)
            b = img[:, :, 2].astype(np.int16)
            # Primary: very dark RGB pixels (black trace)
            black_rgb = (r < 110) & (g < 110) & (b < 110)
            # Secondary: low LAB L (catches dark-grey traces)
            black_lab = L < 75
            black_mask = (black_rgb | black_lab).astype(np.uint8) * 255
            meta["notes"].append(
                f"Color mode: LAB+RGB isolation (pink={pink_ratio*100:.1f}%)"
            )
            meta["is_colour"] = True
        else:
            # Grayscale / B&W mode
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            # Normalize brightness: scale 99th-percentile to 255
            p99 = float(np.percentile(gray, 99))
            if p99 > 30:
                gray = np.clip(
                    gray.astype(float) * (255.0 / p99), 0, 255
                ).astype(np.uint8)
            # CLAHE for local contrast enhancement
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            gray = clahe.apply(gray)
            # Otsu × 1.2 preserves trace peaks (Tereshchenko et al. 2022)
            otsu_val, _ = cv2.threshold(
                gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            thresh = min(int(otsu_val * 1.2), 220)
            _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
            black_mask = bw
            meta["notes"].append(f"Grayscale mode: CLAHE + Otsu×1.2 (thresh={thresh})")
            meta["is_colour"] = False

        # Morphological cleanup: remove isolated noise, close hairline gaps in trace
        open_kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 1))
        cleaned = cv2.morphologyEx(black_mask, cv2.MORPH_OPEN,  open_kernel)
        cleaned = cv2.morphologyEx(cleaned,    cv2.MORPH_CLOSE, close_kernel)

        trace_pixels = int((cleaned > 0).sum())
        meta["notes"].append(
            f"Trace pixels: {trace_pixels} "
            f"({trace_pixels / max(img[:, :, 0].size, 1) * 100:.1f}%)"
        )
        return cleaned

    # ── Layout Detection ───────────────────────────────────────────────────────

    def _detect_layout(
        self, trace_mask: np.ndarray, img_rgb: np.ndarray, meta: dict
    ) -> tuple[list, list]:
        """
        Detect row and column boundaries from trace-pixel projections.

        Rows: find peaks in horizontal projection (dense trace rows).
        Columns: detect high-density vertical separator lines between lead columns.
          Uses the raw (uncleaned) black-pixel mask so thin separator lines are
          not discarded by morphological opening.
        """
        h, w = img_rgb.shape[:2]

        # Horizontal projection (rows) — where are the ECG trace rows?
        h_proj = trace_mask.sum(axis=1).astype(float)   # (h,) — sum per row
        h_smooth = uniform_filter1d(h_proj, size=max(3, h // 40))
        row_bounds = self._find_n_regions(h_smooth, h, n=4, min_regions=3, meta=meta, axis="row")

        # Vertical projection (columns) — detect separator lines between lead columns.
        # Use only the top 3 rows (not the rhythm strip row).
        # Use raw black mask (before morphological cleaning) to preserve thin separators.
        r3_end = row_bounds[min(3, len(row_bounds) - 1)]
        r_ch = img_rgb[:r3_end, :, 0].astype(np.int16)
        g_ch = img_rgb[:r3_end, :, 1].astype(np.int16)
        b_ch = img_rgb[:r3_end, :, 2].astype(np.int16)
        raw_mask = ((r_ch < 120) & (g_ch < 120) & (b_ch < 120)).astype(np.uint8)
        v_proj = raw_mask.sum(axis=0).astype(float)
        col_bounds = self._find_col_bounds_via_separators(v_proj, w, n=4, meta=meta)

        n_rows = len(row_bounds) - 1
        n_cols = len(col_bounds) - 1
        meta["n_rows"] = n_rows
        meta["n_cols"] = n_cols
        meta["notes"].append(f"Layout: {n_rows} rows × {n_cols} cols")
        meta["notes"].append(f"Row bounds: {row_bounds}")
        meta["notes"].append(f"Col bounds: {col_bounds}")
        return row_bounds, col_bounds

    def _find_col_bounds_via_separators(
        self, v_proj: np.ndarray, w: int, n: int, meta: dict
    ) -> list[int]:
        """
        Detect column boundaries by finding vertical separator lines.

        Separator lines between lead columns have pixel density much higher than
        the surrounding ECG trace (typically 5-10× the median). We locate these
        high-density x-positions, cluster them, and use cluster centers as dividers.

        Falls back to peak-based detection, then uniform split if that also fails.
        """
        # Smooth lightly to reduce single-pixel noise
        v_smooth = uniform_filter1d(v_proj, size=max(3, w // 80))

        # Compute median over interior columns (avoid image border effects)
        margin = w // 20
        interior = v_smooth[margin: w - margin]
        if interior.size == 0:
            interior = v_smooth
        med = float(np.median(interior[interior > 0])) if (interior > 0).any() else 1.0

        # Separators are tall vertical lines: density >> median
        # Try progressively lower thresholds until we find n-1 separator clusters
        for threshold in (6.0, 4.0, 3.0, 2.5):
            sep_mask = (v_smooth > threshold * med).astype(np.uint8)
            sep_mask[:margin] = 0
            sep_mask[w - margin:] = 0
            labeled, num = nd_label(sep_mask)
            if num < n - 1:
                continue

            # Get center x of each cluster, sorted left→right
            raw_centers = []
            for idx in range(1, num + 1):
                xs = np.where(labeled == idx)[0]
                # Only count clusters wide enough to be a real separator line
                if len(xs) >= 2:
                    raw_centers.append(int(xs.mean()))
            raw_centers.sort()

            # Merge clusters that are too close together (same separator region)
            # Use w//20 (~40px) so we preserve real separators ~204px apart
            # while merging only truly adjacent pixels from the same line
            min_gap = w // 20
            centers = self._merge_nearby(raw_centers, min_gap)

            if len(centers) >= n - 1:
                # Pick the n-1 most evenly-spaced centers as dividers
                dividers = self._pick_n_dividers(centers, n - 1, w)
                bounds = [0] + dividers + [w]
                meta["notes"].append(
                    f"Col separators detected (thresh={threshold}×median): {dividers}"
                )
                return bounds

        # Fallback 1: peak-based detection (original approach)
        meta["notes"].append("Warning: separator detection failed — trying peak-based col split")
        v_smooth2 = uniform_filter1d(v_proj, size=max(3, w // 40))
        col_bounds = self._find_n_regions(v_smooth2, w, n=n, min_regions=n, meta=meta, axis="col")
        return col_bounds

    def _merge_nearby(self, centers: list[int], min_gap: int) -> list[int]:
        """Merge center positions that are within min_gap pixels of each other."""
        if not centers:
            return centers
        merged = []
        group = [centers[0]]
        for c in centers[1:]:
            if c - group[-1] <= min_gap:
                group.append(c)
            else:
                merged.append(int(np.mean(group)))
                group = [c]
        merged.append(int(np.mean(group)))
        return merged

    def _pick_n_dividers(self, centers: list[int], n: int, total: int) -> list[int]:
        """
        From a list of candidate divider positions, pick n that create the most
        equal-width columns (minimise std of column widths). Tries all combinations.
        """
        import itertools
        if len(centers) <= n:
            return centers[:n]

        best = None
        best_score = float("inf")
        for combo in itertools.combinations(centers, n):
            combo = list(combo)
            widths = [combo[0]] + [combo[i+1] - combo[i] for i in range(n-1)] + [total - combo[-1]]
            score = float(np.std(widths))
            if score < best_score:
                best_score = score
                best = combo
        return best

    def _find_n_regions(
        self,
        profile: np.ndarray,
        total: int,
        n: int,
        min_regions: int,
        meta: dict,
        axis: str = "",
    ) -> list[int]:
        """
        Find n equal-ish regions by locating n peaks in the profile.
        Falls back to uniform split if peak detection fails.
        """
        # Normalize profile
        p = profile / (profile.max() + 1e-6)
        threshold = p.mean() * 0.5

        # Find peaks (dense trace areas)
        from scipy.signal import find_peaks
        peaks, _ = find_peaks(p, height=threshold, distance=total // (n + 1))

        if len(peaks) >= n:
            # Use first n peaks and derive boundaries as midpoints between them
            peaks = peaks[:n]
            bounds = [0]
            for i in range(len(peaks) - 1):
                mid = int((peaks[i] + peaks[i + 1]) / 2)
                bounds.append(mid)
            bounds.append(total)
            return bounds

        # Fallback: find valleys between lead rows (low-density zones)
        low_mask = (p < threshold).astype(np.uint8)
        labeled, num_features = nd_label(low_mask)
        valley_centers = []
        for i in range(1, num_features + 1):
            region = np.where(labeled == i)[0]
            if len(region) > total // (n * 4):  # must be wide enough to be a real gap
                valley_centers.append(int(region.mean()))
        valley_centers.sort()

        if len(valley_centers) >= n - 1:
            bounds = [0] + valley_centers[: n - 1] + [total]
            return bounds

        # Final fallback: uniform split
        meta["notes"].append(f"Warning: uniform {axis} split (detection failed)")
        return [int(i * total / n) for i in range(n + 1)]

    # ── Lead Extraction ────────────────────────────────────────────────────────

    def _extract_all_leads(
        self,
        trace_mask: np.ndarray,
        img_rgb: np.ndarray,
        row_bounds: list,
        col_bounds: list,
        meta: dict,
    ) -> list[np.ndarray]:
        """
        Extract 12 lead waveforms using the hybrid D→B pipeline.
        Returns list of 12 arrays, each resampled to SAMPLES_PER_LEAD (625).
        """
        n_rows = len(row_bounds) - 1
        n_cols = len(col_bounds) - 1
        is_colour = meta.get("is_colour", True)
        leads = [None] * 12

        detected = 0
        for row_idx, col_idx, lead_idx in LAYOUT_3X4:
            if row_idx >= n_rows or col_idx >= n_cols:
                continue

            y0 = row_bounds[row_idx]
            y1 = row_bounds[row_idx + 1]
            x0 = col_bounds[col_idx]
            x1 = col_bounds[col_idx + 1]

            # Small inset to avoid boundary artifacts
            pad_x = max(1, (x1 - x0) // 20)
            pad_y = max(1, (y1 - y0) // 10)
            bin_crop = trace_mask[y0 + pad_y : y1 - pad_y, x0 + pad_x : x1 - pad_x]
            rgb_crop = img_rgb   [y0 + pad_y : y1 - pad_y, x0 + pad_x : x1 - pad_x]

            wave = self._extract_waveform_from_crop(bin_crop, rgb_crop, is_colour)
            leads[lead_idx] = self._resample_signal(wave, SAMPLES_PER_LEAD)
            detected += 1

        meta["n_leads_detected"] = detected

        # Fill missing leads with zeros
        for i in range(12):
            if leads[i] is None:
                leads[i] = np.zeros(SAMPLES_PER_LEAD, dtype=np.float32)

        return leads

    def _extract_waveform_from_crop(
        self,
        bin_crop: np.ndarray,
        rgb_crop: np.ndarray,
        is_colour: bool,
    ) -> np.ndarray:
        """
        Hybrid extraction pipeline (Tereshchenko / Badilini method):

          1. Build the cleanest possible binary for this crop
          2. Remove calibration pulse
          3. Thin to near-single-pixel skeleton (improves Viterbi accuracy)
          4. Viterbi DP path tracing (globally optimal least-cost path)
          5. Fallback to column centroid if Viterbi coverage < 50 %
          6. Smooth + remove baseline wander
        """
        if bin_crop is None or bin_crop.size == 0 or bin_crop.shape[1] < 5:
            return np.zeros(10, dtype=np.float32)

        # ── Step 1: get the best possible binary for this crop ─────────────
        if is_colour:
            working_binary = bin_crop.copy()
        else:
            working_binary = self._grid_subtract_binary(rgb_crop, bin_crop)

        # ── Step 2: remove calibration pulse ──────────────────────────────
        working_binary = self._remove_cal_pulse(working_binary)

        # ── Step 3: skeletonize to single-pixel trace ──────────────────────
        # Reduces multi-pixel-thick traces to single-pixel paths, which
        # dramatically improves the accuracy of the Viterbi path finder.
        thinned = self._thin_trace(working_binary)
        # Use thinned for Viterbi but keep original for centroid fallback
        viterbi_input = thinned if thinned.sum() > 0 else working_binary

        # ── Step 4: Viterbi DP path tracing ───────────────────────────────
        waveform, coverage = self._viterbi_path(viterbi_input)

        # ── Step 5: fallback to column centroid ───────────────────────────
        if coverage < 0.50:
            waveform = self._centroid_extract(working_binary)

        # ── Step 6: smooth (light) + remove baseline wander ───────────────
        if len(waveform) > 20:
            # Light smoothing — preserve QRS morphology
            waveform = uniform_filter1d(waveform, size=max(2, len(waveform) // 80))
        waveform = self._remove_baseline(waveform)
        return waveform.astype(np.float32)

    def _thin_trace(self, binary: np.ndarray) -> np.ndarray:
        """
        Thin a multi-pixel-wide ECG trace to a near-single-pixel skeleton.

        Uses scikit-image Zhang-Suen skeletonization when available; falls back
        to two passes of morphological erosion with an elliptic structuring element.
        """
        if binary is None or binary.sum() == 0:
            return binary
        try:
            from skimage.morphology import skeletonize as sk_thin
            skel = sk_thin(binary > 0).astype(np.uint8) * 255
            if skel.sum() > 0:
                return skel
        except ImportError:
            pass
        # Fallback: one-pass vertical erosion (ECG traces are mostly horizontal)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (1, 3))
        return cv2.erode(binary, kernel, iterations=1)

    # ── Grid subtraction (Method D) ────────────────────────────────────────

    def _grid_subtract_binary(
        self, rgb_crop: np.ndarray, bin_fallback: np.ndarray
    ) -> np.ndarray:
        """
        Remove the ECG grid via morphological opening on the grayscale image.

        The ECG grid is made of long horizontal/vertical lines.  A morphological
        opening with a wide horizontal kernel tracks those lines as the background;
        subtracting them leaves only the shorter, thicker ECG trace signal.
        """
        gray = cv2.cvtColor(rgb_crop, cv2.COLOR_RGB2GRAY).astype(np.float32)
        inv  = 255.0 - gray   # dark trace → bright foreground

        h_k = max(5, rgb_crop.shape[1] // 30)
        v_k = max(3, rgb_crop.shape[0] // 15)

        # Horizontal opening: models horizontal grid lines
        h_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (h_k, 1))
        bg_h = cv2.morphologyEx(inv.astype(np.uint8),
                                cv2.MORPH_OPEN, h_kernel).astype(np.float32)

        # Vertical opening: models vertical grid lines
        v_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, v_k))
        bg_v = cv2.morphologyEx(inv.astype(np.uint8),
                                cv2.MORPH_OPEN, v_kernel).astype(np.float32)

        grid_bg = np.minimum(bg_h, bg_v)
        diff    = np.clip(inv - grid_bg, 0, 255).astype(np.uint8)

        _, binary = cv2.threshold(diff, 0, 255,
                                  cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        # If the result is empty fall back to the original binary
        if binary.sum() == 0:
            return bin_fallback
        return binary

    # ── Calibration pulse removal ──────────────────────────────────────────

    def _remove_cal_pulse(self, binary: np.ndarray) -> np.ndarray:
        """Remove the calibration pulse (tall dense rectangle on right edge)."""
        if binary.shape[1] < 10:
            return binary
        w = binary.shape[1]
        cal_w = max(1, w // 20)
        right_density = binary[:, -cal_w:].mean() / 255.0
        left_density  = binary[:, : w // 2].mean() / 255.0
        if right_density > left_density * 3 and right_density > 0.1:
            binary = binary[:, : w - cal_w]
        return binary

    # ── Viterbi DP path ────────────────────────────────────────────────────

    def _viterbi_path(self, binary: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Dynamic-programming path tracing through candidate trace pixels.

        For each image column we identify the centres of contiguous foreground
        runs (candidate points).  The DP finds the minimum-cost continuous path
        across all columns, balancing Euclidean distance and angular continuity.

        Returns (signal_array, coverage_fraction).  Coverage < 0.50 means the
        binary was too sparse for reliable path-finding → caller should fall back.
        """
        import math

        h, w = binary.shape
        if w == 0:
            return np.zeros(0, dtype=np.float32), 0.0

        DIST_W    = 0.5     # weight: distance vs angle smoothness
        LOOK_BACK = 3       # columns to look back for predecessor

        # candidate points per column: list of y-centres of contiguous runs
        def col_centers(col_pixels):
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

        cols = [col_centers(binary[:, x]) for x in range(w)]
        n_occupied = sum(1 for c in cols if c)

        # not enough signal to trace reliably
        if n_occupied < w * 0.30:
            wave = self._centroid_extract(binary)
            return wave, n_occupied / max(w, 1)

        # dp[x][y] = (total_cost, prev_x, prev_y, angle_in)
        dp = [dict() for _ in range(w)]
        for y in cols[0]:
            dp[0][y] = (0.0, None, None, 0.0)

        def _dist(ax, ay, bx, by):
            return math.sqrt((ax - bx) ** 2 + (ay - by) ** 2)

        def _angle(ax, ay, bx, by):
            dx, dy = bx - ax, by - ay
            if dx == 0 and dy == 0:
                return 0.0
            return math.atan2(dy, dx)

        def _angle_sim(a1, a2):
            diff = abs(a1 - a2)
            if diff > math.pi:
                diff = 2 * math.pi - diff
            return 1.0 - diff / math.pi

        for x in range(1, w):
            for y in cols[x]:
                best_cost = float("inf")
                best_entry = (None, None, 0.0)
                for look in range(1, LOOK_BACK + 1):
                    px = x - look
                    if px < 0:
                        break
                    for py, (pc, _, _, pa) in dp[px].items():
                        ca    = _angle(px, py, x, y)
                        cost  = pc + _dist(px, py, x, y) * DIST_W \
                                   + (1 - _angle_sim(ca, pa)) * (1 - DIST_W)
                        if cost < best_cost:
                            best_cost  = cost
                            best_entry = (px, py, ca)

                # no predecessor found — link to nearest in the previous column
                if best_entry[0] is None and dp[x - 1]:
                    py_near = min(dp[x - 1], key=lambda py: abs(py - y))
                    pc      = dp[x - 1][py_near][0]
                    ca      = _angle(x - 1, py_near, x, y)
                    best_cost  = pc + _dist(x - 1, py_near, x, y)
                    best_entry = (x - 1, py_near, ca)

                dp[x][y] = (best_cost, best_entry[0], best_entry[1], best_entry[2])

        # find best end point in the last 20 columns
        TAIL = min(20, w)
        best_end_cost, end_x, end_y = float("inf"), w - 1, None
        for tx in range(w - TAIL, w):
            for ty, (cost, *_) in dp[tx].items():
                if cost < best_end_cost:
                    best_end_cost, end_x, end_y = cost, tx, ty

        if end_y is None:
            wave = self._centroid_extract(binary)
            return wave, n_occupied / max(w, 1)

        # backtrack
        path = {}
        cx, cy = end_x, end_y
        while cx is not None:
            path[cx] = cy
            _, px, py, _ = dp[cx].get(cy, (0, None, None, 0.0))
            cx, cy = px, py

        # build 1-D signal and invert y-axis
        sig = np.full(w, np.nan)
        for x, y in path.items():
            sig[x] = h - y

        xs = np.where(~np.isnan(sig))[0]
        if len(xs) > 1:
            sig = np.interp(np.arange(w), xs, sig[xs])
        elif len(xs) == 1:
            sig[:] = sig[xs[0]]
        else:
            sig[:] = h / 2.0

        coverage = len(path) / max(w, 1)
        return sig.astype(np.float32), coverage

    # ── Column centroid (fallback) ─────────────────────────────────────────

    def _centroid_extract(self, binary: np.ndarray) -> np.ndarray:
        """Column-by-column centroid of foreground pixels (y inverted)."""
        h, w = binary.shape
        wave = np.full(w, h / 2.0, dtype=np.float64)
        for col in range(w):
            ys = np.where(binary[:, col] > 0)[0]
            if len(ys):
                wave[col] = float(ys.mean())
        return (h - wave).astype(np.float32)

    def _remove_baseline(self, wave: np.ndarray) -> np.ndarray:
        """
        Remove baseline wander using a high-pass filter at 0.5 Hz equivalent.

        The wave is in pixel-space (not true time domain).  We use normalized
        frequency:  cutoff = 0.008 corresponds to removing trends that repeat
        fewer than ~125 times across the signal — i.e. baseline wander.

        A Butterworth 2nd-order high-pass is used (phase-preserving sosfiltfilt).
        """
        if len(wave) < 20:
            return wave
        try:
            # Normalized highpass: removes drifts spanning > ~1/8 of signal length
            cutoff = max(0.005, 8.0 / len(wave))
            sos = signal.butter(2, cutoff, btype="high", fs=1.0, output="sos")
            return signal.sosfiltfilt(sos, wave.astype(float)).astype(np.float32)
        except Exception:
            # Rolling-mean baseline subtraction fallback
            bl = uniform_filter1d(wave.astype(float), size=max(5, len(wave) // 4))
            return (wave - bl).astype(np.float32)

    def _resample_signal(self, wave: np.ndarray, target: int) -> np.ndarray:
        if len(wave) == 0:
            return np.zeros(target, dtype=np.float32)
        if len(wave) == target:
            return wave.astype(np.float32)
        return signal.resample(wave, target).astype(np.float32)

    # ── Waveform Assembly ──────────────────────────────────────────────────────

    def _build_waveform(self, leads: list[np.ndarray], meta: dict) -> np.ndarray:
        """
        Build (2500, 12) waveform array.
        Each lead has 625 samples (2.5 s) → tile 4× → 2500 samples (10 s).
        This preserves heart rate and morphology — the model sees a realistic 10s signal.
        """
        result = np.zeros((TARGET_SAMPLES, 12), dtype=np.float32)
        for i, lead in enumerate(leads):
            if lead is not None and len(lead) > 0:
                # Tile the 2.5-second segment 4 times to simulate 10 seconds
                tiled = np.tile(lead, 4)[:TARGET_SAMPLES]
                result[:, i] = tiled
        meta["notes"].append(f"Leads tiled 4× (2.5s→10s) for model input")
        return result

    # ── Normalization ──────────────────────────────────────────────────────────

    def _normalize_echonext(self, waveform: np.ndarray) -> np.ndarray:
        """
        Per-lead z-score normalization to match EchoNext training distribution.
        Clips at ±5 std (equivalent to EchoNext 0.1/99.9 percentile clipping).
        """
        out = waveform.copy()
        for i in range(out.shape[1]):
            lead = out[:, i]
            std = lead.std()
            if std > 1e-6:
                out[:, i] = np.clip((lead - lead.mean()) / std, -5.0, 5.0)
        return out.astype(np.float32)


# ── Demo ECG (no network required) ────────────────────────────────────────────

def load_demo_ecg() -> tuple[np.ndarray, dict]:
    """Return a synthetic but morphologically realistic normal ECG for demo."""
    fs = 250
    duration = 10
    t = np.linspace(0, duration, fs * duration, dtype=np.float32)
    hr_bpm = 72
    rr = 60.0 / hr_bpm  # seconds per beat

    def make_ecg_lead(t, amplitude=1.0, p_amp=0.15, t_amp=0.3, phase=0.0):
        """Generate a realistic-looking ECG lead using Gaussian pulses."""
        n = len(t)
        ecg = np.zeros(n, dtype=np.float32)
        beat_times = np.arange(0.2, duration, rr)
        for bt in beat_times:
            # P wave
            ecg += p_amp * np.exp(-((t - (bt - 0.16 + phase)) ** 2) / (2 * 0.012**2))
            # Q wave
            ecg -= 0.05 * amplitude * np.exp(-((t - (bt - 0.04)) ** 2) / (2 * 0.004**2))
            # R wave
            ecg += amplitude * np.exp(-((t - bt) ** 2) / (2 * 0.006**2))
            # S wave
            ecg -= 0.15 * amplitude * np.exp(-((t - (bt + 0.04)) ** 2) / (2 * 0.004**2))
            # T wave
            ecg += t_amp * amplitude * np.exp(-((t - (bt + 0.20)) ** 2) / (2 * 0.025**2))
        return ecg

    # Approximate lead amplitudes for a normal ECG
    lead_params = [
        (0.7, 0.15, 0.25),   # I
        (1.0, 0.20, 0.35),   # II
        (0.4, 0.10, 0.15),   # III
        (-0.6, 0.12, 0.20),  # aVR
        (0.3, 0.10, 0.12),   # aVL
        (0.6, 0.15, 0.25),   # aVF
        (-0.3, 0.08, 0.10),  # V1
        (0.5, 0.12, 0.20),   # V2
        (1.0, 0.15, 0.35),   # V3
        (1.4, 0.18, 0.45),   # V4
        (1.2, 0.18, 0.40),   # V5
        (0.9, 0.15, 0.30),   # V6
    ]
    waveform = np.zeros((fs * duration, 12), dtype=np.float32)
    for i, (amp, p, tv) in enumerate(lead_params):
        waveform[:, i] = make_ecg_lead(t, amplitude=amp, p_amp=p, t_amp=tv, phase=i * 0.002)

    # Apply EchoNext normalization
    digitizer = ECGDigitizer()
    waveform = digitizer._normalize_echonext(waveform)

    meta = {
        "success": True,
        "source": "Synthetic normal ECG (72 bpm, normal morphology)",
        "notes": ["No external download required — morphologically realistic demo"],
        "fs": fs,
        "n_leads_detected": 12,
    }
    return waveform, meta
