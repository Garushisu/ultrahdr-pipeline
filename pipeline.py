#!/usr/bin/env python3
"""
NEF to Google Ultra HDR Pipeline (ISO 21496-1)
================================================
Preserves RAW scene-referred linear data, performs high-precision alignment,
true linear exposure merge with Highlight Clipping Suppression,
Auto-Exposure ACES tone mapping with Soft-Knee HDR roll-off,
interactive EXIF selection, robust EXIF preservation via piexif, and Ultra HDR generation via Google libultrahdr.

Author: DeepMind / Antigravity Agent
"""

import os
import sys
import glob
import struct
import argparse
import subprocess
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import cv2
import rawpy
import colour
import piexif

# Enable OpenEXR support in OpenCV
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

# Pipeline Constants
DEFAULT_HDR_BOOST: float = 3.0
TARGET_MIDTONE_LUMA: float = 0.18
MAX_DISPLAY_BOOST: float = 8.0
TARGET_PEAK_NITS: int = 1624
JPEG_QUALITY: int = 95
CLIPPING_SUPPRESSION_THRESHOLD: float = 0.85


def extract_exif_exposure(filepath: str) -> Tuple[float, float]:
    """
    Extract exposure bias (EV) and exposure time (seconds) from NEF EXIF header.
    
    :param filepath: Path to NEF RAW file.
    :return: Tuple of (exposure_bias, exposure_time_seconds).
    """
    try:
        with open(filepath, 'rb') as f:
            data = f.read(500000)
        endian = '<' if data[:2] == b'II' else '>'
        if data[2:4] in (b'\x2a\x00', b'\x00\x2a'):
            ifd0_offset = struct.unpack(endian + 'I', data[4:8])[0]
            tags: Dict[int, Any] = {}
            
            def parse_ifd(offset: int) -> None:
                if offset + 2 > len(data):
                    return
                num_entries = struct.unpack(endian + 'H', data[offset:offset+2])[0]
                curr = offset + 2
                for _ in range(num_entries):
                    if curr + 12 > len(data):
                        break
                    tag, typ, count, val = struct.unpack(endian + 'HHII', data[curr:curr+12])
                    curr += 12
                    if typ in (5, 10):
                        off = val
                        if off + 8 <= len(data):
                            num, den = struct.unpack(endian + ('ii' if typ == 10 else 'II'), data[off:off+8])
                            parsed_val = num / den if den != 0 else 0.0
                        else:
                            parsed_val = 0.0
                    else:
                        parsed_val = float(val)
                    tags[tag] = parsed_val
                    if tag == 0x8769:
                        parse_ifd(val)
                        
            parse_ifd(ifd0_offset)
            bias = float(tags.get(0x9204, 0.0))
            exp = float(tags.get(0x829a, 0.01))
            if exp <= 0:
                exp = 0.01
            return bias, exp
    except Exception:
        pass
    return 0.0, 0.01


def extract_valid_exif_bytes(nef_path: str) -> Optional[bytes]:
    """
    Extracts and validates EXIF payload from a RAW (NEF) file using piexif.
    Returns b'Exif\\x00\\x00' + exif_bytes if valid EXIF exists, or None if missing or corrupted.
    
    :param nef_path: Path to RAW file.
    :return: EXIF APP1 payload bytes or None.
    """
    try:
        if not os.path.exists(nef_path):
            return None
        exif_dict = piexif.load(nef_path)
        
        if not exif_dict.get("0th") and not exif_dict.get("Exif"):
            return None
            
        exif_dict["1st"] = {}
        exif_dict["thumbnail"] = None
        if piexif.ExifIFD.MakerNote in exif_dict.get("Exif", {}):
            del exif_dict["Exif"][piexif.ExifIFD.MakerNote]
            
        exif_bytes = piexif.dump(exif_dict)
        if 0 < len(exif_bytes) <= 65520:
            return b'Exif\x00\x00' + exif_bytes
    except Exception as e:
        print(f"  [EXIF Check] Note: Unable to parse EXIF from {os.path.basename(nef_path)} ({e})")
    return None


def inject_exif_into_jpeg(jpg_path: str, exif_payload: bytes) -> bool:
    """
    Injects EXIF APP1 payload into destination JPEG file right after the SOI marker.
    
    :param jpg_path: Path to JPEG file.
    :param exif_payload: Full EXIF payload (including 'Exif\\x00\\x00' header).
    :return: True if successfully injected, False otherwise.
    """
    try:
        with open(jpg_path, "rb") as f:
            data = f.read()
        if data[:2] != b'\xff\xd8':
            return False

        exif_len = len(exif_payload) + 2
        app1_segment = b'\xff\xe1' + struct.pack('>H', exif_len) + exif_payload

        new_jpeg = b'\xff\xd8' + app1_segment + data[2:]
        with open(jpg_path, "wb") as f:
            f.write(new_jpeg)
        return True
    except Exception as e:
        print(f"  [EXIF Warning] Failed to inject EXIF to {os.path.basename(jpg_path)}: {e}")
        return False


def prompt_select_exif_source(nef_files: List[str], folder_name: str, specified_src: Optional[str] = None) -> Optional[str]:
    """
    Selects EXIF reference file.
    1. If specified_src is provided, uses specified_src.
    2. Auto-detects matching filename (folder_name + '.NEF').
    3. Prompts user interactively using Arrow keys (UP/DOWN) and ENTER if TTY is available.
    """
    if not nef_files:
        return None
        
    if specified_src:
        if specified_src in nef_files:
            return specified_src
        print(f"  [EXIF Warning] Specified file '{specified_src}' not found in folder. Falling back to selection.")

    default_filename = f"{folder_name}.NEF"
    default_idx = 0
    for idx, f in enumerate(nef_files):
        if f.lower() == default_filename.lower():
            default_idx = idx
            break

    if not sys.stdin.isatty():
        selected = nef_files[default_idx]
        print(f"  [EXIF] Non-interactive mode: Auto-selected EXIF source: {selected}")
        return selected

    import tty
    import termios

    print("\n" + "=" * 65)
    print("  📷 Select EXIF Metadata Reference Source File")
    print("  (Use UP / DOWN Arrow Keys to navigate, ENTER to confirm)")
    print("=" * 65)

    current_idx = default_idx
    num_files = len(nef_files)

    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)

    def draw_menu():
        for i, fname in enumerate(nef_files):
            is_def = " (Folder match / Default)" if fname.lower() == default_filename.lower() else ""
            if i == current_idx:
                sys.stdout.write(f"\033[1;36m  ➔ [x] {fname}{is_def}\033[0m\n")
            else:
                sys.stdout.write(f"    [ ] {fname}{is_def}\n")
        sys.stdout.flush()

    try:
        tty.setraw(fd)
        draw_menu()

        while True:
            ch = sys.stdin.read(1)
            if ch == '\x1b':
                ch2 = sys.stdin.read(1)
                if ch2 == '[':
                    ch3 = sys.stdin.read(1)
                    if ch3 == 'A':
                        current_idx = (current_idx - 1) % num_files
                    elif ch3 == 'B':
                        current_idx = (current_idx + 1) % num_files
            elif ch in ('\r', '\n'):
                break
            elif ch == '\x03':
                raise KeyboardInterrupt

            sys.stdout.write(f"\033[{num_files}A")
            draw_menu()
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
        print()

    selected = nef_files[current_idx]
    print(f"  Selected EXIF Reference: {selected}\n")
    return selected


def step1_raw_to_linear(nef_path: str, output_exr: str) -> np.ndarray:
    """
    Step 1: Convert NEF RAW to Scene-Referred Linear sRGB (32-bit float) preserving Camera WB & Warmth.
    
    :param nef_path: Path to input NEF file.
    :param output_exr: Output path for temporary EXR image.
    :return: Linear sRGB Float32 image array (H, W, 3).
    """
    with rawpy.imread(nef_path) as raw:
        linear_srgb_16 = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            output_bps=16,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.sRGB
        )
    
    linear_srgb_float = np.maximum(linear_srgb_16.astype(np.float32) / 65535.0, 0.0)
    cv2.imwrite(output_exr, cv2.cvtColor(linear_srgb_float, cv2.COLOR_RGB2BGR))
    return linear_srgb_float


def step2_align_images(ref_img: np.ndarray, tgt_img: np.ndarray, output_exr: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Step 2: Align target image to reference image using SIFT + ECC in Linear Space.
    """
    h, w = ref_img.shape[:2]
    
    ref_gray = cv2.cvtColor(np.clip(np.power(ref_img, 1/2.2) * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    tgt_gray = cv2.cvtColor(np.clip(np.power(tgt_img, 1/2.2) * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)

    sift = cv2.SIFT_create(nfeatures=5000)
    kp1, des1 = sift.detectAndCompute(ref_gray, None)
    kp2, des2 = sift.detectAndCompute(tgt_gray, None)

    if des1 is None or des2 is None or len(kp1) < 4 or len(kp2) < 4:
        H = np.eye(3, dtype=np.float32)
    else:
        matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)
        matches = matcher.match(des1, des2)
        matches = sorted(matches, key=lambda x: x.distance)[:300]
        if len(matches) >= 4:
            pts1 = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            pts2 = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
            H, _ = cv2.findHomography(pts2, pts1, cv2.RANSAC, 3.0)
            if H is None:
                H = np.eye(3, dtype=np.float32)
        else:
            H = np.eye(3, dtype=np.float32)

    warped_stage1 = cv2.warpPerspective(tgt_img, H, (w, h), flags=cv2.INTER_LANCZOS4)

    try:
        warped_gray = cv2.cvtColor(np.clip(np.power(np.clip(warped_stage1, 0, None), 1/2.2) * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
        _, warp_matrix = cv2.findTransformECC(ref_gray, warped_gray, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        final_aligned = cv2.warpAffine(warped_stage1, warp_matrix, (w, h), flags=cv2.INTER_LANCZOS4)
    except Exception:
        final_aligned = warped_stage1

    mask_in = np.ones((h, w), dtype=np.uint8) * 255
    mask_warped = cv2.warpPerspective(mask_in, H, (w, h), flags=cv2.INTER_NEAREST)
    valid_mask = (mask_warped > 128)

    cv2.imwrite(output_exr, cv2.cvtColor(final_aligned, cv2.COLOR_RGB2BGR))
    return final_aligned, valid_mask


def step3_physically_correct_merge(aligned_images: List[np.ndarray], exposure_times: List[float], output_exr: str) -> np.ndarray:
    """
    Step 3: Exposure-weighted merge for true Scene-Referred HDR.
    Suppresses overexposed/clipped highlight pixels (>0.85) to prevent washout and preserve rich color.
    
    :param aligned_images: List of aligned Float32 RGB images.
    :param exposure_times: List of exposure times in seconds.
    :param output_exr: Output path for merged HDR EXR.
    :return: Merged HDR linear Float32 image.
    """
    h, w, c = aligned_images[0].shape
    hdr_sum = np.zeros((h, w, c), dtype=np.float32)
    weight_sum = np.zeros((h, w, c), dtype=np.float32)
    
    clip_threshold = CLIPPING_SUPPRESSION_THRESHOLD
    
    for img, exp in zip(aligned_images, exposure_times):
        # Gaussian weighting centered at 0.4 for smooth SNR transition
        weight_base = np.exp(-np.square(img - 0.4) / (2.0 * (0.25 ** 2)))
        
        # Highlight Clipping Suppression Mask
        max_c = np.max(img, axis=2, keepdims=True)
        suppression = np.where(
            max_c > clip_threshold,
            np.square(np.clip((1.0 - max_c) / (1.0 - clip_threshold), 0.0, 1.0)),
            1.0
        )
        
        final_weight = weight_base * suppression
        final_weight = np.clip(final_weight, 1e-7, 1.0)
        
        irradiance = img / exp
        hdr_sum += irradiance * final_weight
        weight_sum += final_weight
        
    hdr_merged = hdr_sum / weight_sum
    cv2.imwrite(output_exr, cv2.cvtColor(hdr_merged, cv2.COLOR_RGB2BGR))
    return hdr_merged


def step4_hdr_to_sdr_tonemap(hdr_linear: np.ndarray, ref_exp: float, output_jpg: str) -> Tuple[np.ndarray, float]:
    """
    Step 4: Map Scene-referred HDR to SDR Display using Chroma/Hue-Preserving ACES Filmic + sRGB gamma.
    Preserves exact camera white balance, rich reds, and natural color warmth from original RAW.
    
    :param hdr_linear: Merged HDR Float32 image.
    :param ref_exp: Reference exposure time in seconds.
    :param output_jpg: Output path for base SDR JPEG.
    :return: Tuple of (sdr_8bit_image, auto_exposure_gain).
    """
    scaled = hdr_linear * ref_exp
    luma_in = 0.2126 * scaled[:, :, 0] + 0.7152 * scaled[:, :, 1] + 0.0722 * scaled[:, :, 2]
    
    current_midtone = float(np.median(luma_in))
    if current_midtone > 0:
        auto_gain = TARGET_MIDTONE_LUMA / current_midtone
    else:
        auto_gain = 1.0
        
    auto_gain = float(np.clip(auto_gain, 1.0, 32.0))
    print(f"  [Auto-Exposure] Midtone Luma: {current_midtone:.6f} -> Applying Gain: {auto_gain:.2f}x")
    
    mapped_luma_in = luma_in * auto_gain
    
    def aces_tonemap(x: np.ndarray) -> np.ndarray:
        a = 2.51
        b = 0.03
        c = 2.43
        d = 0.59
        e = 0.14
        return np.clip((x * (a * x + b)) / (x * (c * x + d) + e), 0.0, 1.0)
    
    luma_out = aces_tonemap(mapped_luma_in)
    
    # Calculate Luminance scaling factor to preserve 100% of original RAW RGB ratios (Hue & Saturation)
    denom = np.maximum(mapped_luma_in, 1e-6)
    scale_factor = (luma_out / denom)[:, :, np.newaxis]
    
    sdr_linear = np.clip(scaled * auto_gain * scale_factor, 0.0, 1.0)
    
    # Apply sRGB Gamma Transfer Function
    sdr_gamma = np.where(sdr_linear <= 0.0031308,
                         12.92 * sdr_linear,
                         1.055 * np.power(sdr_linear, 1/2.4) - 0.055)
                         
    sdr_8bit = np.clip(sdr_gamma * 255.0, 0, 255).astype(np.uint8)
    cv2.imwrite(output_jpg, cv2.cvtColor(sdr_8bit, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
    return sdr_8bit, auto_gain


def apply_hdr_highlight_rolloff(hdr_linear_anchored: np.ndarray, max_boost: float = MAX_DISPLAY_BOOST) -> np.ndarray:
    """
    Applies a smooth soft-knee compression curve to the HDR intent signal.
    Prevents hard clipping above 70% of max display boost (e.g. 5.6x up to 8.0x max).
    """
    knee = max_boost * 0.70
    delta = max_boost - knee
    
    res = np.copy(hdr_linear_anchored)
    mask = hdr_linear_anchored > knee
    if np.any(mask):
        over = (hdr_linear_anchored[mask] - knee) / delta
        res[mask] = knee + delta * np.tanh(over)
    return res


def step5_ultrahdr_encode(sdr_jpg_path: str, hdr_linear: np.ndarray, ref_exp: float, auto_gain: float, output_ultrahdr: str, hdr_boost: float = DEFAULT_HDR_BOOST) -> None:
    """
    Step 5: Call Google libultrahdr (ultrahdr_app) to generate ISO 21496-1 Ultra HDR JPEG.
    """
    h, w, _ = hdr_linear.shape
    hdr_rgba = np.ones((h, w, 4), dtype=np.float16)
    
    hdr_anchored = hdr_linear * ref_exp * auto_gain * hdr_boost
    hdr_anchored_rolloff = apply_hdr_highlight_rolloff(hdr_anchored, max_boost=MAX_DISPLAY_BOOST)
    hdr_rgba[:, :, :3] = hdr_anchored_rolloff.astype(np.float16)
    
    tmp_raw_path = os.path.abspath(os.path.join(os.path.dirname(output_ultrahdr), "tmp_hdr_intent.raw"))
    try:
        hdr_rgba.tofile(tmp_raw_path)
        ultrahdr_app_path = os.path.join(os.path.dirname(__file__), "libultrahdr", "build-instagram", "ultrahdr_app")
        
        cmd = [
            ultrahdr_app_path,
            "-m", "0",
            "-p", tmp_raw_path,
            "-i", sdr_jpg_path,
            "-w", str(w),
            "-h", str(h),
            "-a", "4",
            "-C", "2",
            "-t", "0",
            "-z", output_ultrahdr,
            "-M", "0",
            "-K", str(MAX_DISPLAY_BOOST),
            "-L", str(TARGET_PEAK_NITS),
        ]
        
        print("Running libultrahdr:", " ".join(cmd))
        subprocess.run(cmd, check=True)
    finally:
        if os.path.exists(tmp_raw_path):
            os.remove(tmp_raw_path)


def process_folder(input_dir: str, output_dir: str, keep_intermediates: bool = False, hdr_boost: float = DEFAULT_HDR_BOOST, exif_src_file: Optional[str] = None) -> None:
    """
    Process input folder containing NEF RAW exposure stack into Ultra HDR JPEG.
    """
    os.makedirs(output_dir, exist_ok=True)
    folder_name = os.path.basename(os.path.normpath(input_dir))
    
    nef_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.nef')])
    if not nef_files:
        print("No NEF files found.")
        return
        
    selected_exif_file = prompt_select_exif_source(nef_files, folder_name, specified_src=exif_src_file)

    file_info = []
    for f in nef_files:
        p = os.path.join(input_dir, f)
        bias, exp = extract_exif_exposure(p)
        file_info.append({'filename': f, 'path': p, 'bias': bias, 'exp': exp})
        
    ref_item = min(file_info, key=lambda x: abs(x['bias']))
    ref_idx = file_info.index(ref_item)
    print(f"Reference Image: {ref_item['filename']} (Bias: {ref_item['bias']}EV, Exp: {ref_item['exp']}s)")
    
    linear_images = []
    for i, item in enumerate(file_info):
        out_exr = os.path.join(output_dir, f"01_raw_linear_{item['bias']}EV.exr")
        print(f"[{i+1}/5] Step 1: Extracting Linear RAW {item['filename']} -> {out_exr}")
        linear_img = step1_raw_to_linear(item['path'], out_exr)
        linear_images.append(linear_img)
        
    aligned_images = []
    masks = []
    ref_img = linear_images[ref_idx]
    for i, item in enumerate(file_info):
        out_exr = os.path.join(output_dir, f"02_aligned_{item['bias']}EV.exr")
        if i == ref_idx:
            print(f"[{i+1}/5] Step 2: Reference Image (No alignment needed) -> {out_exr}")
            cv2.imwrite(out_exr, cv2.cvtColor(ref_img, cv2.COLOR_RGB2BGR))
            aligned_images.append(ref_img)
            masks.append(np.ones(ref_img.shape[:2], dtype=bool))
        else:
            print(f"[{i+1}/5] Step 2: Aligning {item['filename']} -> {out_exr}")
            alg, mask = step2_align_images(ref_img, linear_images[i], out_exr)
            aligned_images.append(alg)
            masks.append(mask)
            
    combined_mask = np.ones_like(masks[0], dtype=bool)
    for m in masks:
        combined_mask = combined_mask & m
    rows = np.any(combined_mask, axis=1)
    cols = np.any(combined_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    cropped_aligned = [img[rmin:rmax, cmin:cmax] for img in aligned_images]
    
    out_hdr_exr = os.path.join(output_dir, "03_hdr_merged.exr")
    print(f"Step 3: Physically Correct HDR Merge (with Clipping Suppression) -> {out_hdr_exr}")
    exposure_times = [item['exp'] for item in file_info]
    hdr_merged = step3_physically_correct_merge(cropped_aligned, exposure_times, out_hdr_exr)
    
    def print_hdr_statistics(hdr_image: np.ndarray, ref_exp: float) -> None:
        anchored = hdr_image * ref_exp
        lum = 0.2126 * anchored[:, :, 0] + 0.7152 * anchored[:, :, 1] + 0.0722 * anchored[:, :, 2]
        print("\nHDR Statistics (0EV Anchored Luma)")
        print("--------------")
        print(f"Min:   {np.min(lum):.6f}")
        print(f"Max:   {np.max(lum):.6f}")
        print(f"Mean:  {np.mean(lum):.6f}")
        print(f"Median:{np.median(lum):.6f}")
        print(f"P90:   {np.percentile(lum, 90):.6f}")
        print(f"P95:   {np.percentile(lum, 95):.6f}")
        print(f"P99:   {np.percentile(lum, 99):.6f}")
        print(f"P99.9: {np.percentile(lum, 99.9):.6f}\n")

    print_hdr_statistics(hdr_merged, ref_item['exp'])
    
    out_sdr_jpg = os.path.join(output_dir, "04_sdr_tonemapped.jpg")
    print(f"Step 4: ACES Tone Mapping (SDR Generation) -> {out_sdr_jpg}")
    _, auto_gain = step4_hdr_to_sdr_tonemap(hdr_merged, ref_item['exp'], out_sdr_jpg)
    
    out_ultrahdr = os.path.join(output_dir, f"{folder_name}-hdr.jpg")
    print(f"Step 5: Ultra HDR JPEG Generation (libultrahdr, boost={hdr_boost}x) -> {out_ultrahdr}")
    step5_ultrahdr_encode(out_sdr_jpg, hdr_merged, ref_item['exp'], auto_gain, out_ultrahdr, hdr_boost)
    
    if selected_exif_file:
        exif_src_path = os.path.join(input_dir, selected_exif_file)
        exif_payload = extract_valid_exif_bytes(exif_src_path)
        
        if exif_payload is not None:
            print(f"  [EXIF] Valid EXIF metadata detected ({selected_exif_file}). Injecting into Ultra HDR JPEG...")
            inject_exif_into_jpeg(out_ultrahdr, exif_payload)
            if keep_intermediates and os.path.exists(out_sdr_jpg):
                inject_exif_into_jpeg(out_sdr_jpg, exif_payload)
        else:
            print(f"  [EXIF] EXIF metadata is missing or corrupted in {selected_exif_file}. Skipping EXIF injection.")
    else:
        print("  [EXIF] No EXIF reference file selected. Skipping EXIF injection.")

    if not keep_intermediates:
        print("Cleaning up intermediate files...")
        for f in glob.glob(os.path.join(output_dir, "01_raw_linear_*.exr")):
            if os.path.exists(f):
                os.remove(f)
        for f in glob.glob(os.path.join(output_dir, "02_aligned_*.exr")):
            if os.path.exists(f):
                os.remove(f)
        if os.path.exists(out_hdr_exr):
            os.remove(out_hdr_exr)
        if os.path.exists(out_sdr_jpg):
            os.remove(out_sdr_jpg)
            
    print("Done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Ultra HDR Pipeline (ISO 21496-1)")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing NEF files")
    parser.add_argument("-o", "--output", help="Output directory for intermediates and final image (Defaults to input directory)")
    parser.add_argument("--keep_intermediates", action="store_true", help="Keep intermediate EXR and JPG files for debugging")
    parser.add_argument("--hdr_boost", type=float, default=DEFAULT_HDR_BOOST, help=f"Overall brightness multiplier for the HDR intent (default: {DEFAULT_HDR_BOOST})")
    parser.add_argument("--exif_src", help="Specify filename inside input directory to use for EXIF metadata reference")
    args = parser.parse_args()
    
    if args.output is None:
        args.output = args.input
        
    process_folder(args.input, args.output, args.keep_intermediates, args.hdr_boost, args.exif_src)
