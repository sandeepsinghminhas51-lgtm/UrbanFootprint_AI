from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, BackgroundTasks, File, HTTPException, UploadFile, status
from fastapi.responses import FileResponse, JSONResponse

from backend.app.config import (
    RAW_SPACENET_DIR,
    SPACENET_IMAGES_DIR,
    UPLOADS_DIR,
    VALIDATION_METRICS,
    WEIGHTS_PATH,
)
from backend.app.models.schemas import (
    AnalysisRequest,
    BuildingProperty,
    BuildingUpdate,
    JobResultsResponse,
    JobStatusResponse,
    RasterMetadata,
    SampleImageInfo,
)
from backend.app.services.job_service import job_manager
from backend.app.services.vector_service import (
    delete_building_feature,
    load_geojson,
    update_building_property,
)

router = APIRouter()


# ---------------------------------------------------------------------
# HEALTH & MODEL INFO
# ---------------------------------------------------------------------

@router.get("/health")
def get_health() -> Dict[str, Any]:
    return {
        "status": "healthy",
        "service": "UrbanFootprint AI API",
        "model_loaded": WEIGHTS_PATH.exists(),
        "checkpoint_path": str(WEIGHTS_PATH),
    }


@router.get("/model/info")
def get_model_info() -> Dict[str, Any]:
    """Return model architecture, training metadata, and verified validation statistics."""
    return VALIDATION_METRICS


# ---------------------------------------------------------------------
# PRE-LOADED SPACENET SAMPLES (For instant testing)
# ---------------------------------------------------------------------

@router.get("/samples", response_model=List[SampleImageInfo])
def list_sample_images() -> List[SampleImageInfo]:
    """List available real SpaceNet 2 GeoTIFF samples."""
    samples = []
    if SPACENET_IMAGES_DIR.exists():
        for tif in sorted(SPACENET_IMAGES_DIR.glob("*.tif")):
            size_mb = round(tif.stat().st_size / (1024 * 1024), 2)
            # e.g. SN2_buildings_train_AOI_2_Vegas_PS-RGB_img1.tif -> img1
            sample_id = tif.stem.split("_")[-1]
            samples.append(
                SampleImageInfo(
                    id=sample_id,
                    filename=tif.name,
                    display_name=f"SpaceNet 2 Vegas ({sample_id.upper()})",
                    size_mb=size_mb,
                    resolution="0.3m GSD (RGB)",
                    description="High-resolution 3-band Pan-Sharpened RGB GeoTIFF from SpaceNet 2 Building Extraction Challenge (Las Vegas, NV)."
                )
            )
    return samples


@router.post("/samples/load/{sample_id}", response_model=JobStatusResponse)
def load_sample_image(sample_id: str) -> JobStatusResponse:
    """Initialize an analysis job from an existing SpaceNet sample."""
    sample_file = None
    if SPACENET_IMAGES_DIR.exists():
        for tif in SPACENET_IMAGES_DIR.glob(f"*{sample_id}.tif"):
            sample_file = tif
            break

    if not sample_file or not sample_file.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Sample '{sample_id}' not found in {SPACENET_IMAGES_DIR}"
        )

    job_data = job_manager.create_job(sample_file, sample_file.name)
    return JobStatusResponse(**job_data)


# ---------------------------------------------------------------------
# UPLOAD
# ---------------------------------------------------------------------

@router.post("/upload", response_model=JobStatusResponse)
async def upload_satellite_image(file: UploadFile = File(...)) -> JobStatusResponse:
    """
    Upload a georeferenced satellite GeoTIFF (.tif/.tiff).
    Validates file extension and raster readability.
    """
    filename = file.filename or "uploaded_image.tif"
    ext = Path(filename).suffix.lower()

    if ext not in {".tif", ".tiff"}:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file format '{ext}'. Only georeferenced GeoTIFF (.tif / .tiff) files "
                "are accepted. Standard PNG or JPEG files lack geospatial CRS and affine transform "
                "metadata required to compute geographic building coordinates."
            )
        )

    temp_path = UPLOADS_DIR / f"temp_{os.urandom(8).hex()}_{filename}"
    try:
        with open(temp_path, "wb") as f:
            while chunk := await file.read(1024 * 1024 * 4):  # 4MB chunks
                f.write(chunk)

        job_data = job_manager.create_job(temp_path, filename)
        return JobStatusResponse(**job_data)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process uploaded raster: {str(exc)}"
        )
    finally:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except Exception:
                pass


# ---------------------------------------------------------------------
# ANALYSIS & JOB LIFECYCLE
# ---------------------------------------------------------------------

def _run_background_inference(job_id: str, options: AnalysisRequest):
    try:
        job_manager.execute_inference(job_id, options)
    except Exception as e:
        print(f"Error in background inference for job {job_id}: {e}")


@router.post("/analyze/{job_id}", response_model=JobStatusResponse)
def start_analysis(
    job_id: str,
    options: AnalysisRequest,
    background_tasks: BackgroundTasks
) -> JobStatusResponse:
    """Trigger U-Net inference on the uploaded image."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] == "processing":
        return JobStatusResponse(**job)

    # Launch inference as a background task
    background_tasks.add_task(_run_background_inference, job_id, options)

    job["status"] = "processing"
    job["progress"] = 15
    job_manager._save_job_state(job)
    return JobStatusResponse(**job)


@router.get("/jobs", response_model=List[JobStatusResponse])
def list_jobs() -> List[JobStatusResponse]:
    """List recent analysis jobs."""
    jobs = job_manager.list_jobs()
    return [JobStatusResponse(**j) for j in jobs]


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
def get_job_status(job_id: str) -> JobStatusResponse:
    """Get the current processing status and stages for a job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")
    return JobStatusResponse(**job)


@router.get("/jobs/{job_id}/results", response_model=JobResultsResponse)
def get_job_results(job_id: str) -> JobResultsResponse:
    """Get detailed inference results, GIS bounds, and output URLs."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    if job["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed yet (current status: {job['status']})"
        )

    results = job.get("results") or {}
    meta = job.get("metadata") or {}

    bounds_lat_lon = meta.get("wgs84_bounds")
    if not bounds_lat_lon:
        bounds_lat_lon = [[0.0, 0.0], [0.0, 0.0]]

    return JobResultsResponse(
        job_id=job_id,
        status=job["status"],
        metadata=RasterMetadata(**meta),
        building_count=results.get("building_count", 0),
        total_built_area_sqm=results.get("total_built_area_sqm", 0.0),
        avg_building_area_sqm=results.get("avg_building_area_sqm", 0.0),
        threshold_used=results.get("threshold", 0.75),
        inference_time_seconds=results.get("inference_time_seconds", 0.0),
        bounds_lat_lon=bounds_lat_lon,
        preview_image_url=f"/api/jobs/{job_id}/files/preview_satellite.png",
        mask_image_url=f"/api/jobs/{job_id}/files/building_mask.png",
        overlay_image_url=f"/api/jobs/{job_id}/files/building_overlay.png",
        geojson_url=f"/api/jobs/{job_id}/buildings",
        downloads={
            "geojson": f"/api/jobs/{job_id}/export/geojson",
            "mask_tif": f"/api/jobs/{job_id}/download/mask_tif",
            "probability_tif": f"/api/jobs/{job_id}/download/prob_tif",
            "overlay_png": f"/api/jobs/{job_id}/download/overlay_png",
            "summary_json": f"/api/jobs/{job_id}/download/summary",
        }
    )


# ---------------------------------------------------------------------
# GIS & BUILDING MANAGEMENT
# ---------------------------------------------------------------------

@router.get("/jobs/{job_id}/buildings")
def get_buildings_geojson(job_id: str) -> Dict[str, Any]:
    """Retrieve the GeoJSON FeatureCollection for the job."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    geojson_path = Path(job["paths"]["geojson"])
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail="GeoJSON footprints not yet generated.")

    return load_geojson(geojson_path)


@router.patch("/api/jobs/{job_id}/buildings/{building_id}")
@router.patch("/jobs/{job_id}/buildings/{building_id}")
def update_building(
    job_id: str,
    building_id: str,
    update: BuildingUpdate
) -> Dict[str, Any]:
    """Update a building's name or metadata."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    geojson_path = Path(job["paths"]["geojson"])
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail="GeoJSON footprints not found.")

    updated = update_building_property(
        geojson_path=geojson_path,
        building_id=building_id,
        name=update.name,
        notes=update.notes,
        building_type=update.building_type,
    )
    if not updated:
        raise HTTPException(status_code=404, detail=f"Building '{building_id}' not found.")

    return {"status": "success", "updated_building": updated}


@router.delete("/api/jobs/{job_id}/buildings/{building_id}")
@router.delete("/jobs/{job_id}/buildings/{building_id}")
def delete_building(job_id: str, building_id: str) -> Dict[str, Any]:
    """Delete an erroneous building polygon."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    geojson_path = Path(job["paths"]["geojson"])
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail="GeoJSON footprints not found.")

    deleted = delete_building_feature(geojson_path, building_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Building '{building_id}' not found.")

    # Update building count in job state
    data = load_geojson(geojson_path)
    count = len(data.get("features", []))
    if job.get("results"):
        job["results"]["building_count"] = count
        job_manager._save_job_state(job)

    return {"status": "success", "remaining_count": count}


@router.post("/jobs/{job_id}/threshold")
def update_threshold(
    job_id: str,
    body: dict
) -> Dict[str, Any]:
    """Dynamically re-apply a probability threshold using the existing probability raster."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    threshold = float(body.get("threshold", 0.75))
    min_pixels = int(body.get("min_building_pixels", 3))

    try:
        updated = job_manager.reapply_threshold(
            job_id=job_id,
            threshold=threshold,
            min_building_pixels=min_pixels
        )
        return {
            "status": "success",
            "job_id": job_id,
            "threshold": threshold,
            "results": updated["results"],
            "building_count": updated["results"]["building_count"],
            "total_built_area_sqm": updated["results"]["total_built_area_sqm"],
            "geojson": updated["geojson"],
        }
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Failed to apply threshold: {str(exc)}")


@router.get("/jobs/{job_id}/groundtruth")
def get_ground_truth(job_id: str) -> Dict[str, Any]:
    """Retrieve official SpaceNet 2 ground-truth annotations for this scene if available."""
    gt = job_manager.get_ground_truth_geojson(job_id)
    if gt is None:
        return {
            "type": "FeatureCollection",
            "features": [],
            "available": False,
            "message": "Ground-truth vectors not available for this custom upload."
        }
    gt["available"] = True
    return gt


@router.get("/jobs/{job_id}/buildings/{building_id}/export")
def export_single_building(job_id: str, building_id: str) -> JSONResponse:
    """Download a single building footprint as a GeoJSON Feature."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    geojson_path = Path(job["paths"]["geojson"])
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail="GeoJSON footprints not found.")

    data = load_geojson(geojson_path)
    target_feature = None
    for feat in data.get("features", []):
        if feat.get("properties", {}).get("building_id") == building_id or feat.get("id") == building_id:
            target_feature = feat
            break

    if not target_feature:
        raise HTTPException(status_code=404, detail=f"Building '{building_id}' not found.")

    return JSONResponse(
        content=target_feature,
        headers={"Content-Disposition": f"attachment; filename={building_id}.geojson"}
    )


# ---------------------------------------------------------------------
# EXPORT & DOWNLOAD
# ---------------------------------------------------------------------

@router.get("/jobs/{job_id}/export/geojson")
def export_geojson(job_id: str) -> FileResponse:
    """Download the current (potentially edited) GeoJSON file."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    geojson_path = Path(job["paths"]["geojson"])
    if not geojson_path.exists():
        raise HTTPException(status_code=404, detail="GeoJSON file not found.")

    safe_name = Path(job["filename"]).stem
    return FileResponse(
        path=geojson_path,
        media_type="application/geo+json",
        filename=f"{safe_name}_footprints.geojson"
    )


@router.get("/jobs/{job_id}/download/{file_type}")
def download_artifact(job_id: str, file_type: str) -> FileResponse:
    """Download specific output artifacts (mask_tif, prob_tif, overlay_png, summary)."""
    job = job_manager.get_job(job_id)
    if not job:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found.")

    paths = job.get("paths", {})
    # Map friendly aliases
    alias_map = {
        "probability_tif": "prob_tif",
        "mask": "mask_tif",
        "prob": "prob_tif",
        "overlay": "overlay_png",
        "summary_json": "summary",
    }
    actual_key = alias_map.get(file_type, file_type)

    if actual_key not in paths:
        raise HTTPException(status_code=400, detail=f"Invalid file type '{file_type}'.")

    target_path = Path(paths[actual_key])
    if not target_path.exists():
        raise HTTPException(status_code=404, detail=f"Artifact '{file_type}' does not exist.")

    media_type = "application/octet-stream"
    if target_path.suffix == ".png":
        media_type = "image/png"
    elif target_path.suffix in {".tif", ".tiff"}:
        media_type = "image/tiff"
    elif target_path.suffix == ".json":
        media_type = "application/json"

    return FileResponse(
        path=target_path,
        media_type=media_type,
        filename=target_path.name
    )


@router.get("/jobs/{job_id}/files/{filename}")
def serve_job_file(job_id: str, filename: str) -> FileResponse:
    """Serve visual files (preview PNG, overlay PNG, mask PNG) directly to the web UI."""
    job_dir = job_manager.jobs_dir / job_id
    file_path = job_dir / filename

    if not file_path.exists():
        raise HTTPException(status_code=404, detail=f"File '{filename}' not found.")

    media_type = "image/png" if file_path.suffix == ".png" else "application/octet-stream"
    return FileResponse(path=file_path, media_type=media_type)
