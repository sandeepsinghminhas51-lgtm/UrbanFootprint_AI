from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional
import uuid

from backend.app.config import JOBS_DIR, RAW_SPACENET_DIR, SPACENET_IMAGES_DIR, UPLOADS_DIR
from backend.app.models.schemas import (
    AnalysisRequest,
    JobStage,
    JobStatusResponse,
    RasterMetadata,
)
from backend.app.services.inference_service import generate_overlay_image, run_unet_inference
from backend.app.services.raster_service import create_preview_png, inspect_raster
from backend.app.services.vector_service import extract_building_polygons, load_geojson


DEFAULT_STAGES = [
    {"id": "validation", "label": "Image & Geospatial Validation", "status": "pending"},
    {"id": "preprocessing", "label": "Per-Band Percentile Normalization", "status": "pending"},
    {"id": "inference", "label": "U-Net Tiled Inference", "status": "pending"},
    {"id": "mask_generation", "label": "Probability Map & Thresholding", "status": "pending"},
    {"id": "vectorization", "label": "Building Polygon Extraction & WGS84", "status": "pending"},
    {"id": "output_generation", "label": "GeoJSON & Visual Previews", "status": "pending"},
]


class JobManager:
    def __init__(self):
        self.jobs_dir = JOBS_DIR
        self.jobs_dir.mkdir(parents=True, exist_ok=True)

    def _job_file(self, job_id: str) -> Path:
        return self.jobs_dir / job_id / "job.json"

    def _save_job_state(self, job_data: Dict[str, Any]) -> None:
        job_dir = self.jobs_dir / job_data["job_id"]
        job_dir.mkdir(parents=True, exist_ok=True)
        with open(self._job_file(job_data["job_id"]), "w", encoding="utf-8") as f:
            json.dump(job_data, f, indent=2)

    def get_job(self, job_id: str) -> Optional[Dict[str, Any]]:
        job_file = self._job_file(job_id)
        if not job_file.exists():
            return None
        with open(job_file, "r", encoding="utf-8") as f:
            return json.load(f)

    def list_jobs(self, limit: int = 20) -> List[Dict[str, Any]]:
        jobs = []
        for d in self.jobs_dir.iterdir():
            if d.is_dir() and (d / "job.json").exists():
                try:
                    with open(d / "job.json", "r", encoding="utf-8") as f:
                        jobs.append(json.load(f))
                except Exception:
                    continue
        jobs.sort(key=lambda j: j.get("created_at", ""), reverse=True)
        return jobs[:limit]

    def create_job(self, source_path: Path, filename: str) -> Dict[str, Any]:
        job_id = f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
        job_dir = self.jobs_dir / job_id
        job_dir.mkdir(parents=True, exist_ok=True)

        target_tif = job_dir / f"input_{filename}"
        shutil.copy2(source_path, target_tif)

        # Generate raster preview PNG immediately
        preview_png = job_dir / "preview_satellite.png"
        try:
            create_preview_png(target_tif, preview_png)
            preview_url = f"/api/jobs/{job_id}/files/preview_satellite.png"
        except Exception as e:
            print(f"Warning: preview generation error: {e}")
            preview_url = None

        meta = inspect_raster(target_tif, preview_url=preview_url)

        stages = [dict(s) for s in DEFAULT_STAGES]
        stages[0]["status"] = "completed"
        stages[0]["detail"] = f"Validated {meta.width}x{meta.height} px, {meta.count} bands, CRS: {meta.crs}"

        job_data = {
            "job_id": job_id,
            "filename": filename,
            "status": "uploaded",
            "progress": 15,
            "current_stage": "validation",
            "stages": stages,
            "error": None,
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "metadata": meta.model_dump(),
            "paths": {
                "input_tif": str(target_tif),
                "preview_png": str(preview_png),
                "mask_tif": str(job_dir / "building_mask.tif"),
                "prob_tif": str(job_dir / "building_probability.tif"),
                "overlay_png": str(job_dir / "building_overlay.png"),
                "mask_png": str(job_dir / "building_mask.png"),
                "geojson": str(job_dir / "buildings.geojson"),
                "summary": str(job_dir / "prediction_summary.json"),
            },
            "results": None,
        }
        self._save_job_state(job_data)
        return job_data

    def update_stage(
        self,
        job_id: str,
        stage_id: str,
        status: str,
        progress: int,
        detail: Optional[str] = None
    ) -> None:
        job = self.get_job(job_id)
        if not job:
            return
        job["progress"] = progress
        job["current_stage"] = stage_id
        job["updated_at"] = datetime.now().isoformat()
        for stage in job["stages"]:
            if stage["id"] == stage_id:
                stage["status"] = status
                if detail:
                    stage["detail"] = detail
                break
        self._save_job_state(job)

    def execute_inference(self, job_id: str, options: AnalysisRequest) -> Dict[str, Any]:
        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        input_tif = Path(job["paths"]["input_tif"])
        mask_tif = Path(job["paths"]["mask_tif"])
        prob_tif = Path(job["paths"]["prob_tif"])
        overlay_png = Path(job["paths"]["overlay_png"])
        mask_png = Path(job["paths"]["mask_png"])
        geojson_path = Path(job["paths"]["geojson"])
        summary_path = Path(job["paths"]["summary"])

        start_time = time.time()
        job["status"] = "processing"
        self._save_job_state(job)

        try:
            # 1. Validation Stage
            self.update_stage(job_id, "validation", "completed", 15, "Raster geometry and bands verified")

            # 2. Preprocessing Stage
            self.update_stage(job_id, "preprocessing", "in_progress", 20, "Applying 2nd/98th percentile scaling")
            time.sleep(0.1)
            self.update_stage(job_id, "preprocessing", "completed", 25, "Radiometric stretch ready")

            # 3. Inference Stage
            self.update_stage(job_id, "inference", "in_progress", 30, "Running U-Net sliding window inference")

            def progress_cb(pct: float, msg: str):
                self.update_stage(job_id, "inference", "in_progress", int(pct), msg)

            prediction, probabilities, transform, source_crs = run_unet_inference(
                input_path=input_tif,
                output_mask_path=mask_tif,
                output_prob_path=prob_tif,
                threshold=options.threshold,
                patch_size=options.patch_size,
                overlap=options.overlap,
                progress_callback=progress_cb,
            )
            self.update_stage(job_id, "inference", "completed", 75, "U-Net predictions completed")

            # 4. Mask Generation Stage
            self.update_stage(job_id, "mask_generation", "in_progress", 80, f"Applied threshold cutoff: {options.threshold}")
            generate_overlay_image(input_tif, overlay_png, prediction, output_mask_png_path=mask_png)
            self.update_stage(job_id, "mask_generation", "completed", 85, "Generated probability & binary mask")

            # 5. Vectorization Stage
            self.update_stage(job_id, "vectorization", "in_progress", 88, "Extracting polygons & reprojecting to WGS84")
            geojson_data, building_count, total_area = extract_building_polygons(
                prediction=prediction,
                probabilities=probabilities,
                transform=transform,
                source_crs=source_crs,
                output_geojson_path=geojson_path,
                source_name=job["filename"],
                min_area_pixels=options.min_building_pixels,
            )
            self.update_stage(job_id, "vectorization", "completed", 95, f"Extracted {building_count} building footprints")

            # 6. Output Generation
            self.update_stage(job_id, "output_generation", "in_progress", 97, "Finalizing outputs and metadata")
            elapsed = time.time() - start_time

            summary = {
                "job_id": job_id,
                "filename": job["filename"],
                "threshold": options.threshold,
                "building_count": building_count,
                "total_built_area_sqm": round(total_area, 2),
                "avg_building_area_sqm": round(total_area / max(building_count, 1), 2),
                "inference_time_seconds": round(elapsed, 2),
                "predicted_building_pixels": int(prediction.sum()),
                "source_crs": str(source_crs),
                "completed_at": datetime.now().isoformat(),
            }
            with open(summary_path, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            self.update_stage(job_id, "output_generation", "completed", 100, "All outputs generated successfully")

            job = self.get_job(job_id)
            job["status"] = "completed"
            job["progress"] = 100
            job["results"] = summary
            self._save_job_state(job)
            return job

        except Exception as exc:
            import traceback
            traceback.print_exc()
            job = self.get_job(job_id)
            job["status"] = "failed"
            job["error"] = str(exc)
            for s in job["stages"]:
                if s["status"] == "in_progress":
                    s["status"] = "failed"
                    s["detail"] = str(exc)
            self._save_job_state(job)
            raise exc

    def reapply_threshold(
        self,
        job_id: str,
        threshold: float,
        min_building_pixels: int = 3
    ) -> Dict[str, Any]:
        """
        Dynamically apply a new probability threshold without re-running U-Net.
        Re-binarizes the existing probability raster, updates masks and extracts polygons.
        """
        import rasterio
        import numpy as np

        job = self.get_job(job_id)
        if not job:
            raise ValueError(f"Job not found: {job_id}")

        input_tif = Path(job["paths"]["input_tif"])
        mask_tif = Path(job["paths"]["mask_tif"])
        prob_tif = Path(job["paths"]["prob_tif"])
        overlay_png = Path(job["paths"]["overlay_png"])
        mask_png = Path(job["paths"]["mask_png"])
        geojson_path = Path(job["paths"]["geojson"])
        summary_path = Path(job["paths"]["summary"])

        if not prob_tif.exists():
            raise FileNotFoundError(f"Probability raster not found for job: {job_id}. Run initial inference first.")

        # Read existing continuous probability raster
        with rasterio.open(prob_tif) as src:
            probabilities = src.read(1)
            transform = src.transform
            source_crs = src.crs
            profile = src.profile.copy()

        # Binarize with the new threshold
        prediction = (probabilities >= threshold).astype(np.uint8)

        # Update binary mask GeoTIFF
        mask_profile = profile.copy()
        mask_profile.update(
            driver="GTiff",
            count=1,
            dtype="uint8",
            compress="deflate",
            photometric="MINISBLACK",
            nodata=0,
        )
        with rasterio.open(mask_tif, "w", **mask_profile) as dst:
            dst.write(prediction, 1)

        # Update preview PNGs
        generate_overlay_image(input_tif, overlay_png, prediction, output_mask_png_path=mask_png)

        # Re-vectorize building footprints
        geojson_data, building_count, total_area = extract_building_polygons(
            prediction=prediction,
            probabilities=probabilities,
            transform=transform,
            source_crs=source_crs,
            output_geojson_path=geojson_path,
            source_name=job["filename"],
            min_area_pixels=min_building_pixels,
        )

        summary = job.get("results") or {}
        summary["threshold"] = threshold
        summary["building_count"] = building_count
        summary["total_built_area_sqm"] = round(total_area, 2)
        summary["avg_building_area_sqm"] = round(total_area / max(building_count, 1), 2)
        summary["predicted_building_pixels"] = int(prediction.sum())
        summary["updated_at"] = datetime.now().isoformat()

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2)

        job["results"] = summary
        job["updated_at"] = datetime.now().isoformat()
        self._save_job_state(job)

        return {
            "job": job,
            "results": summary,
            "geojson": geojson_data,
        }

    def get_ground_truth_geojson(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Check if SpaceNet 2 official ground truth GeoJSON exists for this scene."""
        job = self.get_job(job_id)
        if not job:
            return None

        filename = job.get("filename", "")
        # Extract image tag (e.g. img1, img3, img10)
        import re
        match = re.search(r"img\d+", filename, re.IGNORECASE)
        if not match:
            return None
        tag = match.group(0).lower()

        labels_dir = RAW_SPACENET_DIR / "labels"
        if not labels_dir.exists():
            return None

        for geojson_file in labels_dir.glob(f"*{tag}.geojson"):
            try:
                with open(geojson_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception:
                pass

        return None


job_manager = JobManager()
