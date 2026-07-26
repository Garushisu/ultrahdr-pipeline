# Ultra HDR Pipeline (ISO 21496-1)

A professional-grade, physically correct HDR pipeline that generates Google Ultra HDR (Gain Map JPEG) from RAW bracketing shots (e.g., Nikon NEF files).

## Architecture

This pipeline is designed from the ground up to preserve RAW sensor data (scene-referred linear light) as much as possible, applying a strictly color-managed workflow using `colour-science` and Google's official `libultrahdr`.

### Processing Steps:
1. **Linear RAW Extraction (`01_raw_linear_*.exr`)**: Reads NEF files using `rawpy`, disables camera curves and white-balance adjustments, and extracts pure Linear XYZ. It then converts this to Linear Rec.2020 (32-bit float) using `colour-science`.
2. **High Precision Alignment (`02_aligned_*.exr`)**: Aligns handheld shots to the 0EV reference image using OpenCV's SIFT (for large homography shifts) and ECC (for sub-pixel exposure-invariant refinement).
3. **Physically Correct HDR Merge (`03_hdr_merged.exr`)**: Computes irradiance (Pixel Value / Exposure Time) and merges the exposures using a linear weighted average (Hat function), strictly preserving Scene-Referred linear values without tone mapping.
4. **HDR to SDR Tone Mapping (`04_sdr_tonemapped.jpg`)**: Applies an ACES Filmic tone mapping curve to compress the high dynamic range into SDR, followed by standard sRGB gamma conversion.
5. **Ultra HDR Generation (`05_final_ultrahdr.jpg`)**: Utilizes Google's official `libultrahdr` (via `ultrahdr_app` CLI) to compute the Gain Map between the HDR intent (Linear Rec.2020) and SDR intent, and encodes an ISO 21496-1 compliant JPEG.

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
   cmake ..
   make -j4
   ```
2. **Install Python Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the pipeline by providing an input directory containing your `.NEF` bracketing files, and an output directory for the results.

```bash
python pipeline.py -i /path/to/NEF/folder -o /path/to/Output/folder
```

### Outputs

The output directory will contain:
- `01_raw_linear_*.exr`: The linear extracted RAW files.
- `02_aligned_*.exr`: The aligned linear files.
- `03_hdr_merged.exr`: The 32-bit float Linear Rec.2020 HDR merged file.
- `04_sdr_tonemapped.jpg`: The ACES tone-mapped SDR base image.
- `05_final_ultrahdr.jpg`: The final Google Ultra HDR Gain Map JPEG.

## Viewing HDR

- **macOS XDR Display**: The generated `.exr` files can be viewed directly in Finder/Preview to experience true 1600 nits hardware HDR.
- **Ultra HDR JPEG**: Drag and drop `05_final_ultrahdr.jpg` into Google Chrome to see the Gain Map HDR effect.
