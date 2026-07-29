# Ultra HDR Pipeline (ISO 21496-1)

A professional-grade, physically correct HDR pipeline that generates Google Ultra HDR (Gain Map JPEG) from RAW bracketing shots (e.g., Nikon NEF files).

## Architecture

This pipeline is designed from the ground up to preserve RAW sensor data (scene-referred linear light) as much as possible, applying a strictly color-managed workflow using `colour-science` and Google's official `libultrahdr`. 

Our V4 architecture features **0EV Physical Anchoring**, guaranteeing mathematically perfect synchronization between the SDR Base and the HDR Intent across any lighting condition.

### Processing Steps:
1. **Linear RAW Extraction (`01_raw_linear_*.exr`)**: Reads NEF files using `rawpy`, disables camera curves and white-balance adjustments, and extracts pure Linear XYZ. It then converts this to Linear Rec.2020 (32-bit float) using `colour-science`.
2. **High Precision Alignment (`02_aligned_*.exr`)**: Aligns handheld shots to the 0EV reference image using OpenCV's SIFT (for large homography shifts) and ECC (for sub-pixel exposure-invariant refinement).
3. **Physically Correct HDR Merge (`03_hdr_merged.exr`)**: Computes absolute irradiance (Pixel Value / Exposure Time) and merges the exposures using a linear weighted average (Hat function), strictly preserving Scene-Referred linear values.
4. **HDR to SDR Tone Mapping (`04_sdr_tonemapped.jpg`)**: 
   - **0EV Anchoring**: Multiplies the absolute irradiance by the 0EV image's exposure time to bring the data back to a relative 0EV scale.
   - Applies an ACES Filmic tone mapping curve to compress the high dynamic range into SDR, followed by standard sRGB gamma conversion.
5. **Ultra HDR Generation (`05_final_ultrahdr.jpg`)**: 
   - The HDR Intent is also scaled to the 0EV anchor, ensuring zero exposure mismatch between SDR and HDR.
   - Utilizes Google's official `libultrahdr` (via `ultrahdr_app` CLI) to compute the Gain Map.
   - Supports adjustable `--hdr_boost` to safely amplify the overall HDR luminance effect without destroying SDR colors.

## Prerequisites

- macOS (or Linux)
- Python 3.10+
- CMake & C++ Compiler (for `libultrahdr` build)

## Setup

1. **Initialize & Build `libultrahdr`**:
   ```bash
   git clone https://github.com/google/libultrahdr.git
   cd libultrahdr
   mkdir build && cd build
   cmake .. \
     -DUHDR_BUILD_EXAMPLES=ON \
     -DUHDR_WRITE_ISO=ON \
     -DUHDR_WRITE_XMP=ON
   make -j4
   ```
2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the pipeline by providing an input directory containing your `.NEF` bracketing files.

```bash
# Basic usage (outputs to the input directory)
python pipeline.py -i /path/to/NEF/folder

# Specify a different output directory and keep intermediate EXR files
python pipeline.py -i /path/to/NEF/folder -o /path/to/Output/folder --keep_intermediates

# Boost the HDR brightness by 8x (default is 4.0)
python pipeline.py -i /path/to/NEF/folder --hdr_boost 8.0
```

### Options
- `-i, --input`: Input directory containing NEF files. (Required)
- `-o, --output`: Output directory. Defaults to the input directory.
- `--keep_intermediates`: Keep intermediate `.exr` and `.jpg` files for debugging. By default, intermediate files are deleted to save space.
- `--hdr_boost`: Overall brightness multiplier applied *only* to the HDR intent (Gain Map). Default is `4.0` (2 stops brighter than SDR). Increase this value (e.g. `8.0`) for a more intense HDR glow on Ultra HDR displays.

## Viewing HDR

- **macOS XDR Display**: The generated `.exr` files (if kept) can be viewed directly in Finder/Preview to experience true hardware HDR.
- **Ultra HDR JPEG**: Drag and drop `05_final_ultrahdr.jpg` into Google Chrome to see the Gain Map HDR effect.
