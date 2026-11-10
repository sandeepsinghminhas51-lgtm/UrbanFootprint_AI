# UrbanFootprint AI — AI-Powered Building Footprint Extraction from Satellite Imagery

**UrbanFootprint AI** is a complete, full-stack geospatial deep-learning platform designed to ingest high-resolution satellite imagery, extract building footprints using a custom U-Net neural network trained from scratch on SpaceNet 2, and provide an interactive GIS map editor for boundary inspection, metadata modification, and GeoJSON export.

---

## 🌟 Key Application Features

1. **Georeferenced Satellite Ingestion**:
   - Drag-and-drop GeoTIFF (`.tif`, `.tiff`) upload with strict metadata validation.
   - Automatic extraction of Coordinate Reference System (CRS), affine geotransform, pixel resolution, and geographic bounding coordinates.
   - Clear explanatory rejection for non-georeferenced images lacking spatial coordinates.

2. **Trained Deep Learning Segmentation**:
   - Executes our custom CPU-compatible U-Net checkpoint (`src/models/weights/urbanfootprint_unet_multisource.pth`).
   - Tiled sliding-window inference ($256 \times 256$ patches) with 2D raised-cosine windowing to eliminate seam artifacts.
   - Continuous probability mapping with configurable confidence threshold (defaulting to the validated optimal threshold of **0.75**).

3. **Sub-Meter GIS Vectorization**:
   - Converts binary prediction masks into planar polygon geometries via 4-connectivity contour tracing.
   - Reprojects all vector coordinates into **WGS84 (EPSG:4326)** for universal web GIS compatibility.
   - Calculates ground footprint area in **square meters ($m^2$)** using latitude-scaled ellipsoidal metric conversion.

4. **Professional 3-Panel GIS Workspace**:
   - Inspired by commercial GIS software like Rescope, with high-resolution satellite imagery as the center of attention.
   - **Left Sidebar**: Multi-layer stack (Base Map, Satellite Raster, Probability Raster, Binary Mask, Vector Polygons, SpaceNet Ground-Truth) and Tools (Select, Pan, Geodesic Distance Measurement).
   - **Dynamic Threshold Tuner**: Interactive slider ($0.30 - 0.90$) with instantaneous re-vectorization using the pre-computed probability map (no U-Net re-inference needed).
   - **Right Sidebar (Feature Inspector)**: Building ID, Name, Confidence, Area in $m^2$ and sq ft, geometry type, and vertex count.
   - **Inline Renaming & Deletion**: Edit names or remove false positives with instant backend persistence (`PATCH`/`DELETE`).
   - **Single & Scene GeoJSON Export**: Export individual features or the complete scene collection.
   - **Live Telemetry Status Bar**: Cursor coordinates (Lat/Lon), raster dimensions, CRS, zoom, and footprint counters.

5. **Model Evaluation Transparency**:
   - Displays real validation metrics evaluated across held-out SpaceNet 2 Las Vegas imagery:
     - **Precision**: 78.43%
     - **Recall**: 84.48%
     - **F1-Score (Dice)**: 81.34%
     - **IoU (Jaccard)**: 68.55%
     - **Selected Optimal Threshold**: 0.75
   - Transparently documents sensor constraints, geographic domain boundaries, and off-nadir angle effects.

---

## 🚀 Quick Start Guide

### Prerequisites
- Python 3.10+ (tested on Python 3.11)
- Node.js 18+ and npm

### 1. Backend Setup & Startup
Open PowerShell in the project root:

```powershell
# Navigate to project root
cd D:\UrbanFootprint_AI

# Start FastAPI backend
.\.venv\Scripts\python.exe backend/run_server.py
```
The backend will launch at `http://127.0.0.1:8000`.
Interactive API documentation is available at `http://127.0.0.1:8000/docs`.

### 2. Frontend Setup & Startup
In a second PowerShell terminal:

```powershell
# Navigate to frontend
cd D:\UrbanFootprint_AI\frontend

# Start Vite development server
npm.cmd run dev
```
The frontend interface will be live at `http://127.0.0.1:5173`.

Alternatively, the frontend production bundle is pre-built in `frontend/dist/` and served directly by the FastAPI backend on `http://127.0.0.1:8000`.

---

## 🧪 Automated Testing

Execute the automated test suite verifying all API endpoints, raster validation, U-Net inference, building renaming, deletion, and GeoJSON export:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_api.py -v
```

All 5 test suites execute against real SpaceNet 2 GeoTIFF data and real model inference:
```
tests/test_api.py::test_health_check PASSED                              [ 20%]
tests/test_api.py::test_model_info PASSED                                [ 40%]
tests/test_api.py::test_samples_listing PASSED                           [ 60%]
tests/test_api.py::test_upload_rejection_for_non_geotiff PASSED          [ 80%]
tests/test_api.py::test_end_to_end_extraction_workflow PASSED            [100%]
```

---

## 📂 Project Architecture

```
UrbanFootprint_AI/
├── backend/
│   ├── app/
│   │   ├── config.py              # Application settings, paths & validation metrics
│   │   ├── main.py                # FastAPI entrypoint, CORS, static SPA routing
│   │   ├── models/schemas.py      # Pydantic data models
│   │   ├── routes/api.py          # REST endpoints (upload, analyze, buildings, export)
│   │   └── services/
│   │       ├── inference_service.py # U-Net model loader & sliding-window inference
│   │       ├── raster_service.py    # Rasterio metadata inspection & RGB preview
│   │       ├── vector_service.py    # Contour extraction & WGS84 area calculation
│   │       └── job_service.py       # Job manager, stage updates & persistence
│   └── run_server.py              # Backend launcher script
│
├── frontend/
│   ├── package.json               # Frontend dependencies & build scripts
│   ├── vite.config.js             # Vite configuration with /api proxy
│   ├── tailwind.config.js         # Custom SaaS theme & color palettes
│   └── src/
│       ├── App.jsx                # Top-level view manager
│       ├── api/client.js          # REST API client
│       └── components/
│           ├── Header.jsx         # App bar with live API health probe
│           ├── Dashboard.jsx      # Metrics overview, test catalog & recent jobs
│           ├── UploadView.jsx     # GeoTIFF upload, metadata inspection & stage tracker
│           ├── GisViewer.jsx      # Leaflet map, multi-layer controls, building editor
│           └── ModelInfoView.jsx  # Architecture details & threshold comparison table
│
├── src/
│   ├── models/                    # Original training, inference & evaluation scripts
│   │   └── weights/               # Trained U-Net checkpoint (.pth)
│   └── postprocessing/            # Polygon vectorization reference logic
│
├── data/
│   ├── raw/spacenet/images/       # Preloaded SpaceNet 2 30cm RGB GeoTIFFs
│   └── processed/                 # Historical predictions & metrics reports
│
├── tests/
│   └── test_api.py                # End-to-end automated pytest suite
│
├── requirements.txt               # Python package dependencies
├── .env.example                   # Environment configuration template
└── README.md                      # Complete project documentation
```

---

## 📜 Scientific Disclaimers & Model Scope
- **Dataset Context**: The model was trained from scratch exclusively on SpaceNet 2 AOI 2 Vegas (30cm RGB imagery).
- **Urban Topology**: Geometric delineation works optimally on single-family suburban and commercial warehouse structures. Complex high-rise urban canyons with significant building facade displacement require specialized off-nadir compensation.
- **Resolution**: Imagery coarser than 0.5m GSD will experience reduced precision.
