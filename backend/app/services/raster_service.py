from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
import rasterio
from rasterio.warp import transform_bounds

from backend.app.models.schemas import RasterMetadata


def calculate_band_limits(src: rasterio.io.DatasetReader, sample_max_side: int = 2048) -> List[Tuple[float, float]]:
    """Calculate 2nd and 98th percentiles per band, matching training preprocessing."""
    scale = min(1.0, sample_max_side / max(src.height, src.width))
    sample_height = max(1, int(round(src.height * scale)))
    sample_width = max(1, int(round(src.width * scale)))

    sample = src.read(
        [1, 2, 3],
        out_shape=(3, sample_height, sample_width),
        masked=True,
    )

    limits = []
    for band_index in range(3):
        band = np.asarray(sample[band_index].compressed(), dtype=np.float32)
        band = band[np.isfinite(band)]
        if band.size == 0:
            raise ValueError(f"RGB band {band_index + 1} contains no valid pixels.")

        low, high = np.percentile(band, [2, 98])
        if not np.isfinite(low) or not np.isfinite(high) or high <= low:
            # Fallback if standard deviation is tiny
            low = float(np.min(band))
            high = float(np.max(band))
            if high <= low:
                high = low + 1.0
        limits.append((float(low), float(high)))

    return limits


def create_preview_png(
    input_path: Path,
    output_png_path: Path,
    max_side: int = 1024
) -> Path:
    """Generate normalized 8-bit RGB PNG preview for web display."""
    output_png_path.parent.mkdir(parents=True, exist_ok=True)

    with rasterio.open(input_path) as src:
        scale = min(1.0, max_side / max(src.height, src.width))
        out_h = max(1, int(round(src.height * scale)))
        out_w = max(1, int(round(src.width * scale)))

        limits = calculate_band_limits(src)
        raw = src.read([1, 2, 3], out_shape=(3, out_h, out_w), masked=True)

        rgb = np.zeros((out_h, out_w, 3), dtype=np.uint8)
        for i, (low, high) in enumerate(limits):
            band = raw[i].astype(np.float32)
            valid = np.isfinite(band)
            scaled = np.clip((band - low) / (high - low), 0.0, 1.0) * 255.0
            scaled[~valid] = 0
            rgb[:, :, i] = scaled.astype(np.uint8)

        img = Image.fromarray(rgb)
        img.save(output_png_path, format="PNG", optimize=True)

    return output_png_path


def inspect_raster(file_path: Path, preview_url: Optional[str] = None) -> RasterMetadata:
    """Inspect GeoTIFF file, extract CRS, bounds, and reproject bounds to WGS84 for Leaflet."""
    with rasterio.open(file_path) as src:
        width = src.width
        height = src.height
        count = src.count
        crs_str = str(src.crs) if src.crs else None
        bounds = [src.bounds.left, src.bounds.bottom, src.bounds.right, src.bounds.top]
        res = [float(src.res[0]), float(src.res[1])]
        dtype = str(src.dtypes[0]) if src.dtypes else "unknown"
        driver = src.driver

        wgs84_bounds = None
        if src.crs:
            try:
                # transform_bounds takes (src_crs, dst_crs, left, bottom, right, top)
                w_left, w_bottom, w_right, w_top = transform_bounds(
                    src.crs, "EPSG:4326", bounds[0], bounds[1], bounds[2], bounds[3]
                )
                # Leaflet format: [[south (lat_min), west (lon_min)], [north (lat_max), east (lon_max)]]
                wgs84_bounds = [
                    [float(w_bottom), float(w_left)],
                    [float(w_top), float(w_right)]
                ]
            except Exception as e:
                print(f"Warning: Could not transform bounds to EPSG:4326: {e}")

        return RasterMetadata(
            filename=file_path.name,
            width=width,
            height=height,
            count=count,
            crs=crs_str,
            bounds=bounds,
            wgs84_bounds=wgs84_bounds,
            resolution=res,
            dtype=dtype,
            driver=driver,
            preview_url=preview_url,
        )
