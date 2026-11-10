from __future__ import annotations

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RasterMetadata(BaseModel):
    filename: str
    width: int
    height: int
    count: int
    crs: Optional[str] = None
    bounds: Optional[List[float]] = None  # [min_x, min_y, max_x, max_y]
    wgs84_bounds: Optional[List[List[float]]] = None  # [[south, west], [north, east]] for Leaflet
    resolution: Optional[List[float]] = None
    dtype: Optional[str] = None
    driver: Optional[str] = None
    preview_url: Optional[str] = None


class JobStage(BaseModel):
    id: str
    label: str
    status: str  # "pending", "in_progress", "completed", "failed"
    detail: Optional[str] = None


class JobStatusResponse(BaseModel):
    job_id: str
    status: str  # "uploaded", "processing", "completed", "failed"
    progress: int  # 0 to 100
    current_stage: str
    stages: List[JobStage]
    error: Optional[str] = None
    created_at: str
    updated_at: str
    metadata: Optional[RasterMetadata] = None


class AnalysisRequest(BaseModel):
    threshold: float = Field(default=0.75, ge=0.1, le=0.95, description="Probability cutoff for building mask")
    overlap: int = Field(default=64, ge=0, le=128, description="Patch overlap in pixels")
    patch_size: int = Field(default=256, ge=128, le=512, description="Inference patch dimension")
    min_building_pixels: int = Field(default=3, ge=1, le=50, description="Minimum area in pixels to keep")


class ThresholdUpdateRequest(BaseModel):
    threshold: float = Field(default=0.75, ge=0.1, le=0.95, description="Probability cutoff for building mask")
    min_building_pixels: int = Field(default=3, ge=1, le=50, description="Minimum area in pixels to keep")


class BuildingUpdate(BaseModel):
    name: Optional[str] = None
    notes: Optional[str] = None
    building_type: Optional[str] = None


class BuildingProperty(BaseModel):
    building_id: str
    name: str
    area_pixels: int
    area_sqm: float
    confidence: float
    status: str  # "ai_detected" or "user_edited"
    source_image: Optional[str] = None
    notes: Optional[str] = None
    building_type: Optional[str] = None


class JobResultsResponse(BaseModel):
    job_id: str
    status: str
    metadata: RasterMetadata
    building_count: int
    total_built_area_sqm: float
    avg_building_area_sqm: float
    threshold_used: float
    inference_time_seconds: float
    bounds_lat_lon: List[List[float]]  # [[south, west], [north, east]]
    preview_image_url: str
    mask_image_url: str
    overlay_image_url: str
    geojson_url: str
    downloads: Dict[str, str]


class SampleImageInfo(BaseModel):
    id: str
    filename: str
    display_name: str
    size_mb: float
    resolution: str
    description: str
