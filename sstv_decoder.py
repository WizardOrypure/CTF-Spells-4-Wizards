#!/usr/bin/env python3
"""
 Note that this is not amazing but should work.

sstv_decoder.py — Auto-detecting SSTV decoder for CTF/radio challenges.

Usage:
    python3 sstv_decoder.py <audio_file> [output_image]

Supports: WAV, MP3, OGG, FLAC (anything ffmpeg can read)
Detects:  Robot36, Robot72, Martin M1/M2, Scottie S1/S2/DX, PD120/PD180

Outputs:
    - Decoded image (PNG)
    - Printed flag candidates from OCR
"""

import sys
import os
import re
import wave
import subprocess
import tempfile
import argparse
import numpy as np
import scipy.signal
from PIL import Image, ImageFilter, ImageEnhance
import cv2

try:
    import pytesseract
    # Verify the tesseract binary is present, not just the Python package.
    pytesseract.get_tesseract_version()
    HAS_OCR = True
except Exception:
    HAS_OCR = False
    print("[!] tesseract not available — OCR disabled. "
          "Install tesseract-ocr and the pytesseract Python package.")


# ─────────────────────────────────────────────────────────────
#  SSTV MODE DEFINITIONS
#  Tuple layout: (name, width, height, line_ms, sync_ms, porch_ms,
#                 scan_ms, sep_ms, sep_porch_ms, c_scan_ms, mode_type)
#
#  VIS code 41 is an alias key for Martin M2 only because Robot72 and Martin M2
#  share the real VIS code 40.  We prefer Robot72 for VIS=40 (more common in
#  practice); Martin M2 is only reached via duration-based guessing, which
#  returns key 41.  Key 41 is never emitted by detect_vis().
# ─────────────────────────────────────────────────────────────
SSTV_MODES = {
    # Robot36: porch is 1.5ms (not 3ms); line=150ms; chroma is half-width at 44ms.
    8:  ("Robot36",    320, 240,  150.0,   9,     1.5,    88.0,   4.5,  1.5,   44.0,   "robot36"),
    # Robot72: porch is 1.5ms; true line is ~198.5ms (not 300ms); Cb and Cr each
    #          take 44ms at FULL width (320px) — both channels transmitted every line.
    40: ("Robot72",    320, 240,  198.5,   9,     1.5,    88.0,   4.5,  1.5,   44.0,   "robot72"),
    41: ("Martin M2",  320, 256,  226.7,   4.862, 0.572,  73.216, 0,    0.572, 73.216, "martin"),
    44: ("Martin M1",  320, 256,  446.4,   4.862, 0.572, 146.432, 0,    0.572,146.432, "martin"),
    60: ("Scottie S1", 320, 256,  428.22,  9,     1.5,   138.24,  0,    1.5,  138.24,  "scottie"),
    56: ("Scottie S2", 320, 256,  277.69,  9,     1.5,    88.064, 0,    1.5,   88.064, "scottie"),
    76: ("Scottie DX", 320, 256, 1050.3,   9,     1.5,   345.6,   0,    1.5,  345.6,   "scottie"),
    95: ("PD120",      640, 496,  508.48, 20,     2.08,  121.6,   0,    0,    121.6,   "pd"),
    96: ("PD180",      640, 496,  754.24, 20,     2.08,  183.04,  0,    0,    183.04,  "pd"),
}


# ─────────────────────────────────────────────────────────────
#  AUDIO LOADING
# ─────────────────────────────────────────────────────────────

def load_audio(path):
    """Load any audio file to mono 44100 Hz float64 array."""
    ext = os.path.splitext(path)[1].lower()

    if ext == ".wav":
        try:
            with wave.open(path) as w:
                rate = w.getframerate()
                nch  = w.getnchannels()
                data = (np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
                          .astype(np.float64) / 32768.0)
            if nch > 1:
                data = data.reshape(-1, nch).mean(axis=1)
            if rate != 44100:
                print(f"[~] Resampling from {rate}Hz to 44100Hz via ffmpeg...")
                return _ffmpeg_resample(path)
            return data, rate
        except Exception as e:
            print(f"[~] wave module failed ({e}), trying ffmpeg...")

    return _ffmpeg_resample(path)


def _ffmpeg_resample(path):
    """Use ffmpeg to decode audio to raw 16-bit mono 44100 Hz PCM."""
    tmp = tempfile.NamedTemporaryFile(suffix=".raw", delete=False)
    tmp.close()
    try:
        cmd = [
            "ffmpeg", "-y", "-i", path,
            "-ac", "1", "-ar", "44100",
            "-f", "s16le", tmp.name,
        ]
        result = subprocess.run(cmd, capture_output=True)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed:\n{result.stderr.decode()}")
        with open(tmp.name, "rb") as f:
            raw = f.read()
    finally:
        os.unlink(tmp.name)

    if len(raw) == 0:
        raise RuntimeError(
            "ffmpeg produced no output — check that the input file contains valid audio.")

    return np.frombuffer(raw, dtype=np.int16).astype(np.float64) / 32768.0, 44100


# ─────────────────────────────────────────────────────────────
#  FM DEMODULATION  (instantaneous frequency via Hilbert)
# ─────────────────────────────────────────────────────────────

def _next_power_of_two(n):
    """Return the smallest power of two >= n."""
    p = 1
    while p < n:
        p <<= 1
    return p


def demodulate(samples, rate, smooth_ms=1.0):
    """Return instantaneous frequency array, smoothed over smooth_ms milliseconds.

    The Hilbert FFT length is rounded up to the next power of two so scipy
    uses a radix-2 FFT, which is significantly faster for long recordings.
    """
    N        = _next_power_of_two(len(samples))
    analytic = scipy.signal.hilbert(samples, N=N)[:len(samples)]
    phase    = np.unwrap(np.angle(analytic))
    freq     = np.diff(phase) / (2.0 * np.pi) * rate
    freq     = np.append(freq, freq[-1])
    window   = max(1, int(smooth_ms / 1000.0 * rate))
    freq     = np.convolve(freq, np.ones(window) / window, mode="same")
    return freq


# ─────────────────────────────────────────────────────────────
#  VIS CODE DETECTION
# ─────────────────────────────────────────────────────────────

def detect_vis(freq, rate):
    """
    Scan for VIS header: ~300ms @ 1900Hz leader, 10ms break @ 1200Hz,
    300ms @ 1900Hz start tone, 10ms start bit @ 1200Hz, then 8 bits at 30ms each.

    Bit 7 of the VIS byte is an even-parity check bit, not a data bit.
    Only bits 0-6 are accumulated into vis_code; bit 7 is validated separately.

    Returns (vis_code, start_of_image_sample) or (None, None).
    """
    LEADER       = 1900.0
    BIT_1        = 1100.0
    BIT_0        = 1300.0
    VIS_TOL      = 100.0   # Hz tolerance — wider to handle SDR frequency drift

    SKIP_MS      = 10 + 300 + 30  # break(10ms) + start tone(300ms) + start bit(30ms) = 340ms

    step          = int(0.010 * rate)   # outer scan step (10ms)
    leader_needed = int(0.250 * rate)   # require at least 250ms of leader tone

    print("[*] Scanning for VIS header...")

    i = 0
    while i < len(freq) - int(1.5 * rate):
        if abs(freq[i] - LEADER) < VIS_TOL:
            # Walk forward to find the end of the leader tone.
            leader_end = i
            while leader_end < len(freq) and abs(freq[leader_end] - LEADER) < VIS_TOL:
                leader_end += 1

            if (leader_end - i) >= leader_needed:
                m       = leader_end + int(SKIP_MS / 1000.0 * rate)
                bit_len = int(0.030 * rate)

                if m + 9 * bit_len >= len(freq):
                    i = leader_end  # skip past, keep scanning
                    continue

                # Read 7 data bits (0-6) and 1 parity bit (7).
                vis_code = 0
                bits     = []
                for b in range(8):
                    start    = m + b * bit_len
                    end      = start + bit_len
                    bit_freq = np.mean(freq[start:end])
                    bit_val  = 1 if abs(bit_freq - BIT_1) < abs(bit_freq - BIT_0) else 0
                    bits.append(bit_val)
                    if b < 7:
                        vis_code |= bit_val << b

                # Validate even parity (warning only — decode proceeds either way).
                parity_bit      = bits[7]
                expected_parity = sum(bits[:7]) % 2  # 0 = even number of 1-bits
                if parity_bit != expected_parity:
                    print(f"[!] VIS parity mismatch (expected {expected_parity}, "
                          f"got {parity_bit}) — possible noise; proceeding anyway")

                img_start = m + 9 * bit_len
                print(f"[+] VIS code detected: {vis_code} @ {i/rate:.2f}s")
                return vis_code, img_start

            # Leader too short — jump past it rather than stepping through slowly.
            i = leader_end
            continue

        i += step

    print("[!] No VIS header found — will use frequency analysis to guess mode")
    return None, None


# ─────────────────────────────────────────────────────────────
#  SYNC PULSE DETECTION
# ─────────────────────────────────────────────────────────────

def find_sync_pulses(freq, rate, line_ms, start_sample=0, n_lines=240,
                     scottie_mode=False):
    """
    Find sync pulse positions for each line, returning an array of sample indices.

    In Scottie modes the sync pulse falls at the END of the previous line period
    (just before the G scan of the current line).  When scottie_mode=True the
    search window is centred on prev_sync + line_samples rather than on a running
    position counter, correctly tracking the shifted Scottie timing.
    """
    line_samples = int(line_ms / 1000.0 * rate)
    syncs        = []
    pos          = start_sample

    for line in range(n_lines):
        expected = (syncs[-1] + line_samples) if (scottie_mode and line > 0) else pos

        search_start = max(0, expected - int(0.020 * rate))
        search_end   = min(len(freq), expected + int(0.050 * rate))
        window       = freq[search_start:search_end]

        # Find the leading edge of the ~1200Hz sync dip.
        min_idx = np.argmin(window)
        if window[min_idx] < 1350:
            pulse_start = min_idx
            while pulse_start > 0 and window[pulse_start - 1] < 1350:
                pulse_start -= 1
            sync_pos = search_start + pulse_start
        else:
            sync_pos = expected  # fallback: use expected position

        syncs.append(sync_pos)
        pos = sync_pos + line_samples

    return np.array(syncs)


# ─────────────────────────────────────────────────────────────
#  PIXEL SAMPLING
# ─────────────────────────────────────────────────────────────

def freq_to_pixel(f):
    """Map SSTV audio frequency to 0-255 pixel value (1500 Hz→0, 2300 Hz→255)."""
    return np.clip((f - 1500.0) / 800.0 * 255.0, 0, 255)


def make_cumsum(freq):
    """Prefix-sum array for O(1) windowed averaging in sample_line."""
    return np.concatenate([[0], np.cumsum(freq)])


def sample_line(freq_cumsum, freq_len, start, duration_ms, n_pixels, rate):
    """Sample n_pixels evenly from the freq window [start, start+duration_ms]."""
    n_samples = int(duration_ms / 1000.0 * rate)
    pixel_w   = max(1, n_samples // n_pixels)
    indices   = start + (np.arange(n_pixels) * n_samples / n_pixels).astype(int)
    i0 = np.clip(indices,           0, freq_len - 1)
    i1 = np.clip(indices + pixel_w, 0, freq_len)
    w  = np.maximum(i1 - i0, 1)
    return freq_to_pixel((freq_cumsum[i1] - freq_cumsum[i0]) / w)


def _u8(arr):
    """Round a float array to the nearest integer and cast to uint8."""
    return np.clip(np.round(arr), 0, 255).astype(np.uint8)


# ─────────────────────────────────────────────────────────────
#  MODE-SPECIFIC IMAGE RECONSTRUCTION
# ─────────────────────────────────────────────────────────────

def decode_robot36(freq, syncs, rate, mode_info):
    _, W, H, line_ms, sync_ms, porch_ms, y_ms, sep_ms, sep_porch_ms, c_ms, _ = mode_info
    print(f"[*] Decoding Robot36: {W}x{H}")
    cs = make_cumsum(freq)
    fl = len(freq)

    img_y  = np.zeros((H, W),    dtype=np.float32)
    img_cr = np.zeros((H, W//2), dtype=np.float32)  # sampled on even lines
    img_cb = np.zeros((H, W//2), dtype=np.float32)  # sampled on odd lines

    for line in range(H):
        s0      = syncs[line]
        y_start = s0 + int((sync_ms + porch_ms) / 1000.0 * rate)
        c_start = y_start + int((y_ms + sep_ms + sep_porch_ms) / 1000.0 * rate)

        img_y[line] = sample_line(cs, fl, y_start, y_ms, W,    rate)
        c_vals      = sample_line(cs, fl, c_start, c_ms, W//2, rate)

        if line % 2 == 0:
            img_cr[line] = c_vals   # even lines carry Cr
        else:
            img_cb[line] = c_vals   # odd lines carry Cb

    # Interpolate unsampled Cr rows (odd lines) from surrounding even rows.
    # Guard uses (i+2) < H because i is odd, so i+1 is the next even (sampled) row.
    for i in range(1, H, 2):
        above = img_cr[i - 1]
        below = img_cr[i + 1] if (i + 2) < H else img_cr[i - 1]
        img_cr[i] = (above + below) / 2.0

    # Interpolate unsampled Cb rows (even lines) from surrounding odd rows.
    for i in range(0, H, 2):
        above = img_cb[i - 1] if i > 0       else img_cb[i + 1]
        below = img_cb[i + 1] if (i + 2) < H else img_cb[i - 1]
        img_cb[i] = (above + below) / 2.0

    return ycbcr_to_rgb(img_y,
                        np.repeat(img_cr, 2, axis=1),
                        np.repeat(img_cb, 2, axis=1))


def decode_robot72(freq, syncs, rate, mode_info):
    _, W, H, line_ms, sync_ms, porch_ms, y_ms, sep_ms, sep_porch_ms, c_ms, _ = mode_info
    print(f"[*] Decoding Robot72: {W}x{H}")
    cs = make_cumsum(freq)
    fl = len(freq)

    img_y  = np.zeros((H, W), dtype=np.float32)
    img_cb = np.zeros((H, W), dtype=np.float32)
    img_cr = np.zeros((H, W), dtype=np.float32)

    for line in range(H):
        s0       = syncs[line]
        y_start  = s0       + int((sync_ms + porch_ms)              / 1000.0 * rate)
        cb_start = y_start  + int((y_ms    + sep_ms + sep_porch_ms) / 1000.0 * rate)
        cr_start = cb_start + int((c_ms    + sep_ms + sep_porch_ms) / 1000.0 * rate)

        # Robot72 transmits Y, Cb, and Cr every line at full horizontal resolution.
        # c_ms (44ms) covers the full 320-pixel width for each chroma channel.
        img_y[line]  = sample_line(cs, fl, y_start,  y_ms, W, rate)
        img_cb[line] = sample_line(cs, fl, cb_start, c_ms, W, rate)
        img_cr[line] = sample_line(cs, fl, cr_start, c_ms, W, rate)

    # ycbcr_to_rgb signature: (Y, Cr, Cb) — Cr is the second argument.
    return ycbcr_to_rgb(img_y, img_cr, img_cb)


def decode_martin(freq, syncs, rate, mode_info):
    _, W, H, line_ms, sync_ms, porch_ms, scan_ms, sep_ms, sep_porch_ms, _, _ = mode_info
    print(f"[*] Decoding Martin: {W}x{H}")
    cs = make_cumsum(freq)
    fl = len(freq)

    R = np.zeros((H, W), dtype=np.float32)
    G = np.zeros((H, W), dtype=np.float32)
    B = np.zeros((H, W), dtype=np.float32)

    for line in range(H):
        s0      = syncs[line]
        g_start = s0      + int((sync_ms + porch_ms)     / 1000.0 * rate)
        b_start = g_start + int((scan_ms + sep_porch_ms) / 1000.0 * rate)
        r_start = b_start + int((scan_ms + sep_porch_ms) / 1000.0 * rate)

        G[line] = sample_line(cs, fl, g_start, scan_ms, W, rate)
        B[line] = sample_line(cs, fl, b_start, scan_ms, W, rate)
        R[line] = sample_line(cs, fl, r_start, scan_ms, W, rate)

    # Stack order [R, G, B] is correct for PIL/numpy RGB.
    # Do NOT pass directly to OpenCV — cv2 expects BGR; use cv2.COLOR_RGB2BGR first.
    return np.stack([_u8(R), _u8(G), _u8(B)], axis=2)


def decode_scottie(freq, syncs, rate, mode_info):
    _, W, H, line_ms, sync_ms, porch_ms, scan_ms, _, sep_porch_ms, _, _ = mode_info
    print(f"[*] Decoding Scottie: {W}x{H}")
    cs = make_cumsum(freq)
    fl = len(freq)

    R = np.zeros((H, W), dtype=np.float32)
    G = np.zeros((H, W), dtype=np.float32)
    B = np.zeros((H, W), dtype=np.float32)

    # Scottie sync is at the END of the previous line period; find_sync_pulses
    # is called with scottie_mode=True so each syncs[line] already points to the
    # sync pulse preceding line's data.  Layout: porch → G → porch → B → porch → R.
    for line in range(H):
        s0      = syncs[line]
        g_start = s0      + int((sync_ms + sep_porch_ms) / 1000.0 * rate)
        b_start = g_start + int((scan_ms + sep_porch_ms) / 1000.0 * rate)
        r_start = b_start + int((scan_ms + sep_porch_ms) / 1000.0 * rate)

        G[line] = sample_line(cs, fl, g_start, scan_ms, W, rate)
        B[line] = sample_line(cs, fl, b_start, scan_ms, W, rate)
        R[line] = sample_line(cs, fl, r_start, scan_ms, W, rate)

    return np.stack([_u8(R), _u8(G), _u8(B)], axis=2)


def decode_pd(freq, syncs, rate, mode_info):
    _, W, H, line_ms, sync_ms, porch_ms, scan_ms, _, _, _, _ = mode_info
    print(f"[*] Decoding PD: {W}x{H}")
    cs = make_cumsum(freq)
    fl = len(freq)

    # PD encodes 2 image lines per sync pulse: Y1 → Cb → Cr → Y2.
    n_syncs = H // 2
    img_y  = np.zeros((H, W),    dtype=np.float32)
    img_cb = np.zeros((H, W//2), dtype=np.float32)
    img_cr = np.zeros((H, W//2), dtype=np.float32)

    for i in range(n_syncs):
        s0       = syncs[i]
        y1_start = s0       + int((sync_ms + porch_ms) / 1000.0 * rate)
        cb_start = y1_start + int(scan_ms              / 1000.0 * rate)
        cr_start = cb_start + int(scan_ms              / 1000.0 * rate)
        y2_start = cr_start + int(scan_ms              / 1000.0 * rate)

        row1, row2 = i * 2, i * 2 + 1
        img_y[row1]  = sample_line(cs, fl, y1_start, scan_ms, W,    rate)
        img_y[row2]  = sample_line(cs, fl, y2_start, scan_ms, W,    rate)
        img_cb[row1] = sample_line(cs, fl, cb_start, scan_ms, W//2, rate)
        img_cb[row2] = img_cb[row1]   # Cb shared between both paired lines
        img_cr[row1] = sample_line(cs, fl, cr_start, scan_ms, W//2, rate)
        img_cr[row2] = img_cr[row1]   # Cr shared between both paired lines

    # ycbcr_to_rgb signature: (Y, Cr, Cb) — Cr is the second argument.
    return ycbcr_to_rgb(img_y,
                        np.repeat(img_cr, 2, axis=1),
                        np.repeat(img_cb, 2, axis=1))


def ycbcr_to_rgb(Y, Cr, Cb):
    """Convert YCbCr planes to an H×W×3 uint8 RGB array (BT.601 coefficients).

    Arguments:
        Y  — luma              (float, 0-255)
        Cr — red-diff chroma   (float, 0-255, 128 = neutral)
        Cb — blue-diff chroma  (float, 0-255, 128 = neutral)
    """
    cr = Cr - 128.0
    cb = Cb - 128.0
    R  = Y + 1.402    * cr
    G  = Y - 0.344136 * cb - 0.714136 * cr
    B  = Y + 1.772    * cb
    return np.stack([_u8(R), _u8(G), _u8(B)], axis=2)


DECODERS = {
    "robot36": decode_robot36,
    "robot72": decode_robot72,
    "martin":  decode_martin,
    "scottie": decode_scottie,
    "pd":      decode_pd,
}


# ─────────────────────────────────────────────────────────────
#  GUESS MODE BY AUDIO DURATION
# ─────────────────────────────────────────────────────────────

def guess_mode_from_duration(n_samples, rate):
    """Pick the most likely SSTV mode based on total audio duration."""
    duration = n_samples / rate
    print(f"[*] Audio duration: {duration:.1f}s — guessing mode from duration...")

    # Key 41 is the alias for Martin M2 (see SSTV_MODES comment above).
    candidates = {
        8:  36,    # Robot36    ≈ 36s   (240 lines × 150ms)
        40: 48,    # Robot72    ≈ 48s   (240 lines × 198.5ms)
        41: 58,    # Martin M2  ≈ 58s
        44: 114,   # Martin M1  ≈ 114s
        60: 110,   # Scottie S1 ≈ 110s
        56: 71,    # Scottie S2 ≈ 71s
        95: 240,   # PD120      ≈ 240s
        96: 360,   # PD180      ≈ 360s
    }

    best_code = min(candidates, key=lambda c: abs(duration - candidates[c]))
    print(f"[~] Best guess: {SSTV_MODES[best_code][0]} "
          f"(expected ~{candidates[best_code]}s, got {duration:.1f}s)")
    return best_code


# ─────────────────────────────────────────────────────────────
#  AUTO-DETECT IMAGE START
# ─────────────────────────────────────────────────────────────

def find_image_start(freq, rate, line_ms):
    """Find image start by locating the first coherent sequence of sync pulses."""
    line_samples = int(line_ms / 1000.0 * rate)
    step         = int(0.005 * rate)  # 5ms scan step

    print(f"[*] Searching for sync pulse train (line_ms={line_ms:.1f})...")

    for i in range(0, max(0, len(freq) - line_samples * 5), step):
        if freq[i] < 1350:
            hits = sum(
                1 for k in range(1, 8)
                if freq[
                    max(0, i + k * line_samples - int(0.020 * rate)):
                    min(len(freq), i + k * line_samples + int(0.020 * rate))
                ].min() < 1350
            )
            if hits >= 5:
                print(f"[+] Sync train found at {i/rate:.3f}s ({hits}/7 confirmations)")
                return i

    print("[!] Could not find sync train — starting from sample 0")
    return 0


# ─────────────────────────────────────────────────────────────
#  OCR  — find flag candidates
# ─────────────────────────────────────────────────────────────

# Matches common CTF flag formats: PREFIX{body} where prefix may contain hyphens
# and body is 3-128 non-whitespace, non-brace characters.
_FLAG_RE       = re.compile(r'[A-Za-z0-9][A-Za-z0-9_\-]{0,15}\{[^}\s]{3,128}\}')
_FLAG_KEYWORDS = frozenset(("FLAG", "CTF", "KEY", "SECRET", "HACK", "HTB", "THM"))


def extract_flags(img_array):
    """Run OCR on the image with multiple preprocessings, return sorted flag candidates."""
    if not HAS_OCR:
        return []

    img_pil    = Image.fromarray(img_array)
    candidates = set()

    def ocr_pass(img):
        try:
            text = pytesseract.image_to_string(img, config="--psm 6")
        except Exception as e:
            print(f"[!] OCR pass failed: {e}")
            return
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            for match in _FLAG_RE.findall(line):
                candidates.add(match)
            if any(kw in line.upper() for kw in _FLAG_KEYWORDS):
                candidates.add(line)

    gray = img_pil.convert("L")

    ocr_pass(img_pil)                                           # Pass 1: raw RGB
    ocr_pass(gray.filter(ImageFilter.SHARPEN))                  # Pass 2: sharpened grey
    ocr_pass(ImageEnhance.Contrast(gray).enhance(3.0))         # Pass 3: high contrast
    ocr_pass(gray.point(lambda x: 255 if x > 128 else 0, "1")) # Pass 4: binary threshold

    # Pass 5: adaptive threshold via OpenCV
    cv_img    = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2GRAY)
    cv_thresh = cv2.adaptiveThreshold(cv_img, 255,
                                      cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    ocr_pass(Image.fromarray(cv_thresh))

    return sorted(candidates)


# ─────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────

def decode_sstv(input_path, output_path=None):
    if output_path is None:
        base        = os.path.splitext(input_path)[0]
        output_path = base + "_decoded.png"

    print(f"\n{'='*55}")
    print(f"  SSTV Decoder")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    print(f"{'='*55}\n")

    # 1. Load audio
    print("[*] Loading audio...")
    samples, rate = load_audio(input_path)
    print(f"[+] Loaded {len(samples)/rate:.1f}s @ {rate}Hz mono")

    # 2. Demodulate
    print("[*] Demodulating...")
    freq = demodulate(samples, rate, smooth_ms=1.0)

    # 3. Detect VIS code
    vis_code, img_start_sample = detect_vis(freq, rate)

    # 4. Resolve mode
    if vis_code is not None and vis_code in SSTV_MODES:
        mode_info = SSTV_MODES[vis_code]
        print(f"[+] Mode: {mode_info[0]}")
    else:
        if vis_code is not None:
            print(f"[!] Unknown VIS code {vis_code}, falling back to duration guess")
        mode_info = SSTV_MODES[guess_mode_from_duration(len(samples), rate)]
        # Discard img_start from an unknown VIS hit — likely a false positive.
        img_start_sample = None

    W, H, line_ms = mode_info[1], mode_info[2], mode_info[3]
    mode_type     = mode_info[10]

    # 5. Find image start if VIS didn't provide it
    if img_start_sample is None:
        img_start_sample = find_image_start(freq, rate, line_ms)

    # 6. Find sync pulse positions
    # PD encodes 2 image lines per sync pulse, so only H//2 syncs exist in the stream.
    n_syncs    = H // 2 if mode_type == "pd" else H
    is_scottie = mode_type == "scottie"
    print(f"[*] Finding {n_syncs} line syncs{' (Scottie mode)' if is_scottie else ''}...")
    syncs = find_sync_pulses(freq, rate, line_ms, img_start_sample, n_syncs,
                             scottie_mode=is_scottie)
    print(f"[+] Sync range: {syncs[0]/rate:.3f}s → {syncs[-1]/rate:.3f}s")

    # 7. Decode image
    rgb = DECODERS[mode_type](freq, syncs, rate, mode_info)
    Image.fromarray(rgb, "RGB").save(output_path)
    print(f"\n[+] Image saved: {output_path}")

    # 8. OCR for flags
    print("\n[*] Running OCR for flag candidates...")
    flags = extract_flags(rgb)
    if flags:
        print("\n" + "="*55)
        print("  *** POSSIBLE FLAGS ***")
        print("="*55)
        for f in flags:
            print(f"  >> {f}")
        print("="*55)
    else:
        print("[~] No flag-like text detected by OCR.")
        print("    Check the image manually — the flag may be purely graphical.")

    print(f"\n[✓] Done! Open: {output_path}")
    return output_path, flags


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Auto-detecting SSTV decoder")
    parser.add_argument("input",  help="Input audio file (WAV, MP3, OGG, FLAC, ...)")
    parser.add_argument("output", nargs="?", help="Output PNG path (optional)")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"[!] File not found: {args.input}")
        sys.exit(1)

    decode_sstv(args.input, args.output)
