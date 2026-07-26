#!/usr/bin/env python3
"""
NEF to Google Ultra HDR Pipeline (ISO 21496-1)
Preserves RAW scene-referred linear data, performs high-precision alignment,
true linear exposure merge, ACES tone mapping, and Ultra HDR generation via libultrahdr.
"""

import os
import sys
import re
import struct
import argparse
import subprocess
import numpy as np
import cv2
import rawpy
import colour

# Ensure OpenCV can read/write EXR
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"

def get_exif_data(filepath):
    """Extract exposure bias and exposure time (shutter speed) from NEF EXIF header."""
    try:
        with open(filepath, 'rb') as f:
            data = f.read(500000)
        endian = '<' if data[:2] == b'II' else '>'
        if data[2:4] in (b'\x2a\x00', b'\x00\x2a'):
            ifd0_offset = struct.unpack(endian + 'I', data[4:8])[0]
            tags = {}
            def parse_ifd(offset):
                if offset + 2 > len(data): return
                num_entries = struct.unpack(endian + 'H', data[offset:offset+2])[0]
                curr = offset + 2
                for _ in range(num_entries):
                    if curr + 12 > len(data): break
                    tag, typ, count, val = struct.unpack(endian + 'HHII', data[curr:curr+12])
                    curr += 12
                    if typ in (5, 10):
                        off = val
                        if off + 8 <= len(data):
                            num, den = struct.unpack(endian + ('ii' if typ==10 else 'II'), data[off:off+8])
                            parsed_val = num / den if den != 0 else 0
                        else: parsed_val = 0
                    else: parsed_val = val
                    tags[tag] = parsed_val
                    if tag == 0x8769: parse_ifd(val)
            parse_ifd(ifd0_offset)
            bias = tags.get(0x9204, 0.0)
            exp = tags.get(0x829a, 0.01)
            if exp <= 0: exp = 0.01
            return bias, exp
    except Exception:
        pass
    return 0.0, 0.01

def step1_raw_to_linear(nef_path, output_exr):
    """
    Step 1: Convert NEF to Scene-Referred Linear Rec.2020 (32-bit float)
    """
    with rawpy.imread(nef_path) as raw:
        # Extract purely linear data, output in XYZ color space to avoid camera sRGB curves
        linear_xyz_16 = raw.postprocess(
            gamma=(1, 1),
            no_auto_bright=True,
            output_bps=16,
            use_camera_wb=True,
            output_color=rawpy.ColorSpace.XYZ
        )
    
    # Convert from 16-bit to Float32 [0.0, 1.0]
    linear_xyz_float = linear_xyz_16.astype(np.float32) / 65535.0
    
    # Convert XYZ to Linear Rec.2020 using colour-science
    # rawpy XYZ output illuminant is usually D50
    illuminant_XYZ = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D50']
    
    linear_rec2020 = colour.XYZ_to_RGB(
        linear_xyz_float,
        colour.models.RGB_COLOURSPACE_BT2020,
        illuminant=illuminant_XYZ,
        chromatic_adaptation_transform='CAT02'
    )
    
    # Clip negative values that can result from matrix conversion, but keep highlights
    linear_rec2020 = np.maximum(linear_rec2020, 0.0).astype(np.float32)
    
    # Save as EXR (BGR order required for OpenCV)
    cv2.imwrite(output_exr, cv2.cvtColor(linear_rec2020, cv2.COLOR_RGB2BGR))
    return linear_rec2020

def step2_align_images(ref_img, tgt_img, output_exr):
    """
    Step 2: Align images using SIFT + ECC in Linear Space.
    Images are 32-bit float RGB.
    """
    h, w = ref_img.shape[:2]
    
    # Scale to 8-bit for feature detection (apply simple gamma to help SIFT find features)
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
            if H is None: H = np.eye(3, dtype=np.float32)
        else:
            H = np.eye(3, dtype=np.float32)

    # Warp float32 target image
    warped_stage1 = cv2.warpPerspective(tgt_img, H, (w, h), flags=cv2.INTER_LANCZOS4)

    try:
        warped_gray = cv2.cvtColor(np.clip(np.power(np.clip(warped_stage1, 0, None), 1/2.2) * 255.0, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
        warp_matrix = np.eye(2, 3, dtype=np.float32)
        criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 100, 1e-5)
        _, warp_matrix = cv2.findTransformECC(ref_gray, warped_gray, warp_matrix, cv2.MOTION_TRANSLATION, criteria)
        final_aligned = cv2.warpAffine(warped_stage1, warp_matrix, (w, h), flags=cv2.INTER_LANCZOS4)
    except Exception:
        final_aligned = warped_stage1

    # Generate mask
    mask_in = np.ones((h, w), dtype=np.uint8) * 255
    mask_warped = cv2.warpPerspective(mask_in, H, (w, h), flags=cv2.INTER_NEAREST)
    valid_mask = (mask_warped > 128)

    cv2.imwrite(output_exr, cv2.cvtColor(final_aligned, cv2.COLOR_RGB2BGR))
    return final_aligned, valid_mask

def step3_physically_correct_merge(aligned_images, exposure_times):
    """
    Step 3: Exposure weighted merge for true Scene-Referred HDR.
    Inputs are Linear Float32 RGB.
    Returns absolute irradiance.
    """
    h, w, c = aligned_images[0].shape
    hdr_sum = np.zeros((h, w, c), dtype=np.float32)
    weight_sum = np.zeros((h, w, c), dtype=np.float32)
    
    for img, exp in zip(aligned_images, exposure_times):
        weight = 1.0 - np.abs(np.clip(img, 0.0, 1.0) - 0.5) * 2.0
        weight = np.clip(weight, 1e-6, 1.0)
        irradiance = img / exp
        hdr_sum += irradiance * weight
        weight_sum += weight
        
    hdr_merged = hdr_sum / weight_sum
    return hdr_merged

def normalize_and_cap_hdr(hdr_absolute, percentile=95.0, max_gain=8.0):
    """
    Normalize HDR absolute irradiance to 1.0 based on percentile, 
    and cap max highlights.
    """
    # Luminance for Rec.2020
    lum = 0.2627 * hdr_absolute[:,:,0] + 0.6780 * hdr_absolute[:,:,1] + 0.0593 * hdr_absolute[:,:,2]
    p_val = np.percentile(lum, percentile)
    
    hdr_norm = hdr_absolute / (p_val + 1e-6)
    hdr_intent = np.clip(hdr_norm, 0.0, max_gain)
    return hdr_intent

def sdr_soft_knee_tonemap_luminance(hdr_intent, knee=0.8):
    """
    Step 4: Map HDR Intent to SDR using Luminance-Ratio Soft-Knee.
    Preserves chromaticity (no color shift) and guarantees Gain=1.0 for darks/mids.
    """
    # 1. Luminance (Rec.2020)
    Y = 0.2627 * hdr_intent[:,:,0] + 0.6780 * hdr_intent[:,:,1] + 0.0593 * hdr_intent[:,:,2]
    
    # 2. Soft-Knee on Luminance ONLY
    Y_sdr = np.copy(Y)
    mask = Y > knee
    Y_sdr[mask] = knee + (1.0 - knee) * (1.0 - np.exp(-(Y[mask] - knee) / (1.0 - knee)))
    
    # 3. Ratio
    ratio = Y_sdr / (Y + 1e-6)
    
    # 4. Apply to RGB
    sdr_linear = hdr_intent * ratio[:, :, np.newaxis]
    return sdr_linear

def step4_sdr_generation(sdr_linear, output_jpg):
    """
    Convert SDR Linear Rec.2020 to sRGB Gamma JPEG.
    """

    
    # Convert Linear Rec.2020 SDR back to Linear sRGB for standard JPEG
    illuminant_XYZ = colour.CCS_ILLUMINANTS['CIE 1931 2 Degree Standard Observer']['D65']
    sdr_linear_srgb = colour.RGB_to_RGB(
        sdr_linear,
        colour.models.RGB_COLOURSPACE_BT2020,
        colour.models.RGB_COLOURSPACE_sRGB,
        chromatic_adaptation_transform='CAT02'
    ).astype(np.float32)
    sdr_linear_srgb = np.clip(sdr_linear_srgb, 0.0, 1.0)
    
    # Apply sRGB Gamma Transfer Function
    sdr_gamma = np.where(sdr_linear_srgb <= 0.0031308,
                         12.92 * sdr_linear_srgb,
                         1.055 * np.power(sdr_linear_srgb, 1/2.4) - 0.055)
                         
    sdr_8bit = np.clip(sdr_gamma * 255.0, 0, 255).astype(np.uint8)
    
    # Save base JPEG
    cv2.imwrite(output_jpg, cv2.cvtColor(sdr_8bit, cv2.COLOR_RGB2BGR), [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    return sdr_8bit

def step5_ultrahdr_encode(sdr_jpg_path, hdr_linear, output_ultrahdr):
    """
    Step 5: Call Google libultrahdr (ultrahdr_app)
    Requires generating a temporary raw RGBA Half-Float binary for HDR intent.
    """
    h, w, c = hdr_linear.shape
    
    # Convert HDR Linear Rec.2020 to RGBA Half Float (float16)
    # Alpha channel must be 1.0
    hdr_rgba = np.ones((h, w, 4), dtype=np.float16)
    hdr_rgba[:, :, :3] = hdr_linear.astype(np.float16)
    
    tmp_raw_path = "tmp_hdr_intent.raw"
    hdr_rgba.tofile(tmp_raw_path)
    
    ultrahdr_app_path = os.path.join(os.path.dirname(__file__), "libultrahdr", "build", "ultrahdr_app")
    
    cmd = [
        ultrahdr_app_path,
        "-m", "0",                          # Encode mode
        "-p", tmp_raw_path,                 # HDR intent raw file
        "-i", sdr_jpg_path,                 # SDR intent compressed jpeg
        "-w", str(w),
        "-h", str(h),
        "-a", "4",                          # rgbaHalfFloat
        "-C", "2",                          # bt2100 (Rec.2020) gamut
        "-t", "0",                          # Linear transfer function
        "-z", output_ultrahdr
    ]
    
    print("Running libultrahdr:", " ".join(cmd))
    subprocess.run(cmd, check=True)
    
    if os.path.exists(tmp_raw_path):
        os.remove(tmp_raw_path)

def process_folder(input_dir, output_dir, args):
    os.makedirs(output_dir, exist_ok=True)
    
    nef_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith('.nef')])
    if not nef_files:
        print("No NEF files found.")
        return
        
    file_info = []
    for f in nef_files:
        p = os.path.join(input_dir, f)
        bias, exp = get_exif_data(p)
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
            
    # Crop to valid bounds (Intersection of masks)
    combined_mask = np.ones_like(masks[0], dtype=bool)
    for m in masks: combined_mask = combined_mask & m
    rows = np.any(combined_mask, axis=1)
    cols = np.any(combined_mask, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    
    cropped_aligned = [img[rmin:rmax, cmin:cmax] for img in aligned_images]
    
    print("Step 3: Physically Correct HDR Merge (Absolute Irradiance)")
    exposure_times = [item['exp'] for item in file_info]
    hdr_absolute = step3_physically_correct_merge(cropped_aligned, exposure_times)
    
    out_hdr_exr = os.path.join(output_dir, "03_hdr_intent.exr")
    print(f"Step 3b: Percentile Normalization (p={args.percentile}) & Capping (max={args.max_gain}) -> {out_hdr_exr}")
    hdr_intent = normalize_and_cap_hdr(hdr_absolute, args.percentile, args.max_gain)
    cv2.imwrite(out_hdr_exr, cv2.cvtColor(hdr_intent, cv2.COLOR_RGB2BGR))
    
    out_sdr_jpg = os.path.join(output_dir, "04_sdr_tonemapped.jpg")
    print(f"Step 4: SDR Tone Mapping (Luminance Soft-Knee) -> {out_sdr_jpg}")
    sdr_linear = sdr_soft_knee_tonemap_luminance(hdr_intent, knee=0.8)
    step4_sdr_generation(sdr_linear, out_sdr_jpg)
    
    out_ultrahdr = os.path.join(output_dir, "05_final_ultrahdr.jpg")
    print(f"Step 5: Ultra HDR JPEG Generation (libultrahdr) -> {out_ultrahdr}")
    step5_ultrahdr_encode(out_sdr_jpg, hdr_intent, out_ultrahdr)
    
    print("Done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Google Ultra HDR Pipeline (ISO 21496-1)")
    parser.add_argument("-i", "--input", required=True, help="Input directory containing NEF files")
    parser.add_argument("-o", "--output", default="Output", help="Output directory for intermediates and final image")
    parser.add_argument("--percentile", type=float, default=95.0, help="Percentile for exposure normalization (e.g. 95)")
    parser.add_argument("--max_gain", type=float, default=8.0, help="Maximum HDR gain restriction (e.g. 8.0)")
    args = parser.parse_args()
    
    process_folder(args.input, args.output, args)
