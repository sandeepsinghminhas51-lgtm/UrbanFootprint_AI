# UrbanFootprint AI — Frontend & GIS Viewer Documentation

## 1. Overview
The **UrbanFootprint AI** frontend is a modern, responsive Single Page Application (SPA) designed with a professional SaaS dashboard aesthetic. Built with **React 18**, **Vite**, **Tailwind CSS**, and **Leaflet**, it gives geospatial analysts and machine learning practitioners an interactive environment to ingest satellite imagery, monitor deep learning segmentation pipelines, explore extracted vector building footprints, and perform in-place GIS polygon editing.

---

## 2. Directory Structure
```
frontend/
├── index.html               # Main entry HTML with Leaflet CSS & Google Fonts
├── package.json             # NPM package manifests & scripts
├── postcss.config.js        # PostCSS configuration
├── tailwind.config.js       # Modern Tailwind theme (brand colors, fonts, slate scale)
├── vite.config.js           # Vite config with backend proxy (/api -> :8000)
├── dist/                    # Production bundle output
└── src/
    ├── main.jsx             # React DOM root render
    ├── App.jsx              # Application state machine & active view router
    ├── index.css            # Tailwind directives, Leaflet tweaks, scrollbars
    ├── api/
    │   └── client.js        # REST client communicating with FastAPI backend
    └── components/
        ├── Header.jsx       # App branding, live API status ping, navigation tabs
        ├── Dashboard.jsx    # SaaS hero, validated metrics widget, test catalog, job history
        ├── UploadView.jsx   # Drag-and-drop GeoTIFF ingestion, metadata cards, stage progress
        ├── GisViewer.jsx    # Interactive Leaflet GIS workspace, layers, building management
        └── ModelInfoView.jsx # Architecture breakdown, threshold sensitivity, scientific limitations
```

---

## 3. Key Components & Capabilities

### 3.1 Dashboard (`Dashboard.jsx`)
- **Hero & Quick Action**: Instant entry point with one-click SpaceNet 2 sample loading.
- **Model Validation Report**: Transparently displays empirical metrics ($F1=81.34\%$, $\text{IoU}=68.55\%$, $\text{Precision}=78.43\%$, $\text{Recall}=84.48\%$).
- **Test Imagery Catalog**: Direct access to workspace SpaceNet 2 Pan-Sharpened RGB GeoTIFFs.
- **Recent Analysis History**: Lists past extraction runs with status, timestamp, and footprint counts.

### 3.2 Ingestion & Analysis Pipeline (`UploadView.jsx`)
- **Drag & Drop Upload**: Validates `.tif` / `.tiff` extensions. Automatically warns the user if non-georeferenced imagery (PNG/JPG) is uploaded without spatial reference.
- **Pre-Analysis Metadata**: Visualizes image dimensions, spectral band counts, Coordinate Reference System (CRS), pixel resolution, and geographic bounding coordinates.
- **Inference Tuner**: Interactive sliders for threshold cutoff (default 0.75), patch overlap (64px), and minimum polygon area (3px).
- **Stage Progress Tracker**: Live progress bar and checklist monitoring all 6 execution stages in real-time.

### 3.3 Three-Panel Commercial GIS Workspace (`GisViewer.jsx`)
Inspired directly by high-end geospatial property analysis tools like Rescope:
- **Left Collapsible Sidebar (Layers & Tools)**:
  - **Layers Tab**: Controls for Base Map, Satellite GeoTIFF, AI Probability Raster, Binary Building Mask, Extracted Vectors, and SpaceNet Ground-Truth.
  - **Tools Tab**: Select Tool, Free Pan Canvas, and Interactive Geodesic Distance Measurement Tool (meters).
  - **Dynamic Threshold Control**: Real-time slider ($0.30 - 0.90$) with instantaneous re-vectorization via the pre-calculated probability raster (no U-Net re-inference needed).
- **Center Canvas (Interactive Map)**:
  - High-resolution georeferenced satellite overlay with terracotta/orange vector building boundaries.
  - Floating zoom and fit-coverage toolbar.
- **Right Collapsible Sidebar (Feature Inspector)**:
  - Overview table: Building ID, Name, Confidence %, Footprint Area ($m^2$ and sq ft), Geometry type, vertex count.
  - Properties & Centroid: Lat/Lon coordinates, threshold used, analysis scene name.
  - Inline editing & polygon deletion with live backend synchronization.
  - Individual building GeoJSON export and complete scene GeoJSON export.
  - Searchable directory of scene building footprints.
- **Bottom Telemetry Status Bar**:
  - Live cursor coordinates (Lat, Lon), raster dimensions, CRS, zoom level, detected footprint count, and active threshold.

### 3.4 Model Intelligence (`ModelInfoView.jsx`)
- Detailed architectural schematic of the custom U-Net.
- Full threshold sensitivity comparison table across held-out SpaceNet 2 validation images.
- Clear disclosure of sensor constraints, geographic domain limitations, and off-nadir angle effects.

---

## 4. Development & Build Commands

### Start Vite Development Server
```powershell
cd frontend
npm.cmd run dev
```
Development server will be live at `http://127.0.0.1:5173`.

### Compile Production Bundle
```powershell
cd frontend
npm.cmd run build
```
Build output is saved to `frontend/dist/` and automatically served by FastAPI when running in production mode.
