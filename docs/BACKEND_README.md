# UrbanFootprint AI — Backend & Inference Engine Documentation

## 1. Overview
The **UrbanFootprint AI** backend is a high-performance Python application built on **FastAPI**. It dynamically integrates with our custom SpaceNet 2 U-Net deep learning model (`urbanfootprint_unet_multisource.pth`), processing high-resolution satellite GeoTIFFs, computing probability arrays, extracting vector polygons, and serving interactive GIS data to the web frontend.

---

## 2. Directory Structure
```
backend/
├── app/
│   ├── config.py              # Central application configuration & paths
│   ├── main.py                # FastAPI app initialization, CORS, static file mounts
│   ├── models/
│   │   └── schemas.py         # Pydantic schemas for requests, responses & metadata
│   ├── routes/
│   │   └── api.py             # REST API endpoint handlers
│   └── services/
│       ├── inference_service.py # U-Net model loader, tiled sliding-window inference
│       ├── raster_service.py    # Rasterio metadata inspection & RGB preview creation
│       ├── vector_service.py    # 4-connectivity polygon vectorization & WGS84 transform
│       └── job_service.py       # Job lifecycle, persistent state & stage tracker
└── run_server.py              # CLI launcher for the Uvicorn development server
```

---

## 3. Deep Learning & Geospatial Pipeline

### 3.1 Model Architecture
- **Type**: Custom 4-stage U-Net encoder-decoder with residual-style DoubleConv blocks.
- **Base Channels**: 16 (`BASE_CHANNELS = 16`), scaling to 32, 64, 128 (bottleneck).
- **Execution Target**: CPU-optimized (`torch.device("cpu")`), capped at 4 threads to prevent lock contention.
- **Input Channels**: 3 (Pan-Sharpened Red, Green, Blue bands).
- **Weights File**: `src/models/weights/urbanfootprint_unet_multisource.pth`.

### 3.2 Radiometric Normalization
To match the exact distribution used during training (`prepare_multi_image_dataset.py`):
1. Bands 1, 2, and 3 are sampled.
2. The 2nd and 98th percentile pixel intensities are calculated for each band.
3. Intensities are scaled to the range $[0.0, 1.0]$.
4. Edge tiles are zero-padded to $256 \times 256$ pixels.

### 3.3 Tiled Inference with Cosine Blending
To eliminate artificial boundary seams between adjacent inference patches:
- **Patch Size**: $256 \times 256$ pixels.
- **Overlap**: 64 pixels (configurable).
- **Window Weighting**: 2D raised-cosine weight matrix applied across overlapping patch regions.
- **Output**: Continuous floating-point probability map $[0.0, 1.0]$.
- **Binarization**: Threshold cutoff applied at $0.75$ (selected from empirical validation).

### 3.4 Vectorization & Area Calculation
1. **Contour Extraction**: `rasterio.features.shapes` with 4-connectivity extracts connected building pixels.
2. **Noise Filtration**: Polygons smaller than 3 pixels are filtered.
3. **Reprojection**: `rasterio.warp.transform_geom` converts pixel coordinates to **WGS84 (EPSG:4326)**.
4. **Metric Area ($m^2$)**: Computed using latitude-scaled ellipsoidal metric conversion:
   $$\text{Scale}_{\text{lon}} = \frac{\pi}{180} \cdot 6,378,137 \cdot \cos(\text{lat})$$
   $$\text{Scale}_{\text{lat}} \approx 111,132.95 - 559.82 \cdot \cos(2\cdot\text{lat})$$
5. **Feature Assembly**: Each building receives a persistent ID (`building_00001`), customizable name, confidence score, centroid coordinates, and is packaged into a valid GeoJSON `FeatureCollection`.

---

## 4. REST API Specification

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/health` | Health probe & model checkpoint verification |
| `GET` | `/api/model/info` | Architecture details & verified validation statistics |
| `GET` | `/api/samples` | List available pre-loaded SpaceNet 2 GeoTIFF samples |
| `POST` | `/api/samples/load/{sample_id}` | Initialize a job from an existing SpaceNet sample |
| `POST` | `/api/upload` | Upload a new georeferenced GeoTIFF (`.tif`/`.tiff`) |
| `POST` | `/api/analyze/{job_id}` | Start U-Net building footprint extraction in background |
| `GET` | `/api/jobs` | List recent analysis jobs |
| `GET` | `/api/jobs/{job_id}` | Current progress percentage and stage statuses |
| `GET` | `/api/jobs/{job_id}/results` | Extraction summary metrics, GIS bounds & URLs |
| `GET` | `/api/jobs/{job_id}/buildings` | GeoJSON `FeatureCollection` of extracted footprints |
| `PATCH` | `/api/jobs/{job_id}/buildings/{building_id}` | Edit building name / notes (persisted to GeoJSON) |
| `DELETE` | `/api/jobs/{job_id}/buildings/{building_id}` | Remove false positive polygon from GeoJSON |
| `POST` | `/api/jobs/{job_id}/threshold` | Re-binarize and vectorize via existing probability raster |
| `GET` | `/api/jobs/{job_id}/groundtruth` | SpaceNet 2 official human ground-truth GeoJSON (if scene exists) |
| `GET` | `/api/jobs/{job_id}/buildings/{building_id}/export` | Export individual selected building polygon as GeoJSON |
| `GET` | `/api/jobs/{job_id}/export/geojson` | Download updated, user-edited GeoJSON file |
| `GET` | `/api/jobs/{job_id}/download/{file_type}` | Download artifacts (`mask_tif`, `prob_tif`, `overlay_png`, `summary`) |
| `GET` | `/api/jobs/{job_id}/files/{filename}` | Serve raster preview / mask PNG for web visualization |

---

## 5. Running the Backend

Ensure the virtual environment is activated, then run:

```powershell
.\.venv\Scripts\python.exe backend/run_server.py
```

The API will be accessible at:
- **API Base URL**: `http://127.0.0.1:8000`
- **Interactive Swagger Docs**: `http://127.0.0.1:8000/docs`
