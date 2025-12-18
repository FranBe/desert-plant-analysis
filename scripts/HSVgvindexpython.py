"""
hsvgvi.py

Implementation of HSVGVI image construction from:
Song et al., "A New Remote Sensing Desert Vegetation Detection Index" (Remote Sens. 2023).
Implements the paper's RGB->HSV conversion, S/V enhancement (×1.15), HSV->RGB reconstruction,
and channel mixing / green enhancement (Eq. 6) to produce the HSVGVI image.

References in paper: Equations (3)-(6). See paper for details. :contentReference[oaicite:3]{index=3} :contentReference[oaicite:4]{index=4}
"""

import numpy as np
from PIL import Image
from typing import Tuple, Union


def load_image_as_array(path: str) -> np.ndarray:
    """Load image from path and return uint8 HxWx3 RGB array."""
    print("Este es el path: ")
    print(path)
    img = Image.open(path).convert("RGB")
    return np.asarray(img, dtype=np.uint8)


def rgb_to_hsv_custom(rgb: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Convert uint8 RGB (0-255) to H (degrees 0-360), S (0-1), V (0-1)
    Using the formulas in the paper (Eq.3 and Eq.4). Returns H (float32),
    S (float32), V (float32), all shaped HxW.
    """
    if rgb.dtype != np.float32 and rgb.dtype != np.float64:
        rgb = rgb.astype(np.float32)

    R = rgb[..., 0] / 255.0
    G = rgb[..., 1] / 255.0
    B = rgb[..., 2] / 255.0

    Cmax = np.maximum(np.maximum(R, G), B)
    Cmin = np.minimum(np.minimum(R, G), B)
    delta = Cmax - Cmin

    # Hue H in degrees [0,360)
    H = np.zeros_like(Cmax, dtype=np.float32)

    # delta == 0 -> H = 0 (paper)
    mask = delta != 0

    # When max == R
    mask_r = (Cmax == R) & mask
    H[mask_r] = 60.0 * (((G[mask_r] - B[mask_r]) / delta[mask_r]) % 6)

    # When max == G
    mask_g = (Cmax == G) & mask
    H[mask_g] = 60.0 * (((B[mask_g] - R[mask_g]) / delta[mask_g]) + 2.0)

    # When max == B
    mask_b = (Cmax == B) & mask
    H[mask_b] = 60.0 * (((R[mask_b] - G[mask_b]) / delta[mask_b]) + 4.0)

    # Saturation S
    S = np.zeros_like(Cmax, dtype=np.float32)
    nonzero_max = Cmax != 0
    S[nonzero_max] = delta[nonzero_max] / Cmax[nonzero_max]
    # V is Cmax
    V = Cmax.astype(np.float32)

    # Ensure ranges
    H = np.mod(H, 360.0)  # handle negative mod
    S = np.clip(S, 0.0, 1.0)
    V = np.clip(V, 0.0, 1.0)

    return H.astype(np.float32), S.astype(np.float32), V.astype(np.float32)


def hsv_enhance_and_convert_to_rgb(H: np.ndarray, S: np.ndarray, V: np.ndarray,
                                   sv_boost: float = 1.15) -> np.ndarray:
    """
    Boost S and V by sv_boost (e.g., 1.15 per paper), then convert HSV -> RGB.
    Return uint8 RGB image (HxWx3) in 0..255.
    Conversion follows Eq. (5) of the paper.
    """
    # Boost S and V
    S2 = S * sv_boost
    V2 = V * sv_boost

    # Clip S and V to valid ranges
    S2 = np.clip(S2, 0.0, 1.0)
    V2 = np.clip(V2, 0.0, 1.0)

    C = V2 * S2
    H_div_60 = H / 60.0
    # compute X = C * (1 - |(H/60 mod 2) - 1|)
    # We can compute the fractional part mod 2 by H_div_60 % 2
    mod2 = np.mod(H_div_60, 2.0)
    X = C * (1.0 - np.abs(mod2 - 1.0))
    m = V2 - C

    # Prepare empty channels
    R_p = np.zeros_like(H, dtype=np.float32)
    G_p = np.zeros_like(H, dtype=np.float32)
    B_p = np.zeros_like(H, dtype=np.float32)

    # Assign according to H range
    # 0 ≤ H < 60 -> (C, X, 0)
    mask = (H >= 0) & (H < 60)
    R_p[mask], G_p[mask], B_p[mask] = C[mask], X[mask], 0.0

    # 60 ≤ H < 120 -> (X, C, 0)
    mask = (H >= 60) & (H < 120)
    R_p[mask], G_p[mask], B_p[mask] = X[mask], C[mask], 0.0

    # 120 ≤ H < 180 -> (0, C, X)
    mask = (H >= 120) & (H < 180)
    R_p[mask], G_p[mask], B_p[mask] = 0.0, C[mask], X[mask]

    # 180 ≤ H < 240 -> (0, X, C)
    mask = (H >= 180) & (H < 240)
    R_p[mask], G_p[mask], B_p[mask] = 0.0, X[mask], C[mask]

    # 240 ≤ H < 300 -> (X, 0, C)
    mask = (H >= 240) & (H < 300)
    R_p[mask], G_p[mask], B_p[mask] = X[mask], 0.0, C[mask]

    # 300 ≤ H < 360 -> (C, 0, X)
    mask = (H >= 300) & (H < 360)
    R_p[mask], G_p[mask], B_p[mask] = C[mask], 0.0, X[mask]

    # Add m and scale to 0..255
    R = np.clip((R_p + m) * 255.0, 0.0, 255.0).astype(np.uint8)
    G = np.clip((G_p + m) * 255.0, 0.0, 255.0).astype(np.uint8)
    B = np.clip((B_p + m) * 255.0, 0.0, 255.0).astype(np.uint8)

    rgb_out = np.stack([R, G, B], axis=-1)
    return rgb_out


def compute_hsvgvi_from_rgb_array(rgb_uint8: np.ndarray, sv_boost: float = 1.15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Main pipeline:
      - input: HxWx3 uint8 RGB image (0..255)
      - RGB -> HSV (H degrees, S 0..1, V 0..1) [Eq.3-4]
      - enhance S and V by sv_boost and convert to RGB (HSVVI step: Eq.5)
      - form HSVGVI image by channel mixing (Eq.6): R' = R_hsv * G_hsv, G' = G_hsv * 2, B' = B_hsv
    Returns:
      - hsvgvi_float: HxWx3 float image (0..1)
      - hsvgvi_uint8: HxWx3 uint8 image (0..255)
    """
    # Convert to HSV using the paper's formula
    H, S, V = rgb_to_hsv_custom(rgb_uint8)

    # Convert enhanced HSV back to RGB (HSVVI)
    hsvvi_rgb_uint8 = hsv_enhance_and_convert_to_rgb(H, S, V, sv_boost=sv_boost)

    # Convert HSVVI rgb units to float 0..1 for channel mixing
    hsvvi_rgb_f = hsvvi_rgb_uint8.astype(np.float32) / 255.0
    R_h = hsvvi_rgb_f[..., 0]
    G_h = hsvvi_rgb_f[..., 1]
    B_h = hsvvi_rgb_f[..., 2]

    # Channel mixing per paper Eq. (6): R_new = R_h * G_h ; G_new = G_h * 2 ; B_new = B_h
    R_new = R_h * G_h
    G_new = G_h * 2.0
    B_new = B_h

    # Clip to [0,1]
    R_new = np.clip(R_new, 0.0, 1.0)
    G_new = np.clip(G_new, 0.0, 1.0)
    B_new = np.clip(B_new, 0.0, 1.0)

    hsvgvi_float = np.stack([R_new, G_new, B_new], axis=-1)
    hsvgvi_uint8 = (np.clip(hsvgvi_float, 0.0, 1.0) * 255.0).astype(np.uint8)

    return hsvgvi_float, hsvgvi_uint8


def save_uint8_image(arr_uint8: np.ndarray, path: str):
    """Save HxWx3 uint8 array to path."""
    img = Image.fromarray(arr_uint8, mode="RGB")
    img.save(path)


# Example usage:
if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("Usage: python hsvgvi.py INPUT_RGB_IMAGE OUTPUT_HSVGVI_IMAGE")
        print("Example: python hsvgvi.py uav_image.png hsvgvi_out.png")
        sys.exit(1)

    inp = sys.argv[1]
    out = sys.argv[2]
    rgb = load_image_as_array(inp)
    _, hsvgvi_u8 = compute_hsvgvi_from_rgb_array(rgb, sv_boost=1.15)
    save_uint8_image(hsvgvi_u8, out)
    print(f"HSVGVI image saved to {out}")
