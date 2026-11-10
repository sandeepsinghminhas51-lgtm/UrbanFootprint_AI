from __future__ import annotations

import sys
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.app.main import app

client = TestClient(app)


def test_health_check():
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True


def test_model_info():
    response = client.get("/api/model/info")
    assert response.status_code == 200
    data = response.json()
    assert data["optimal_threshold"] == 0.75
    assert data["combined_validation"]["f1"] == 0.8134
    assert data["combined_validation"]["iou"] == 0.6855
    assert "SpaceNet 2" in data["dataset"]


def test_samples_listing():
    response = client.get("/api/samples")
    assert response.status_code == 200
    samples = response.json()
    assert isinstance(samples, list)
    assert len(samples) > 0
    assert any("img1" in s["filename"] for s in samples)


def test_upload_rejection_for_non_geotiff():
    fake_content = b"This is a non-geotiff text file"
    response = client.post(
        "/api/upload",
        files={"file": ("test.png", fake_content, "image/png")}
    )
    assert response.status_code == 400
    detail = response.json()["detail"]
    assert "Only georeferenced GeoTIFF (.tif / .tiff) files are accepted" in detail


def test_end_to_end_extraction_workflow():
    # 1. Load SpaceNet sample 1
    load_res = client.post("/api/samples/load/img1")
    assert load_res.status_code == 200
    job_data = load_res.json()
    job_id = job_data["job_id"]
    assert job_data["metadata"]["crs"] is not None
    assert len(job_data["metadata"]["wgs84_bounds"]) == 2

    # 2. Start U-Net Analysis with optimal threshold 0.75
    analysis_res = client.post(
        f"/api/analyze/{job_id}",
        json={"threshold": 0.75, "overlap": 64, "patch_size": 256, "min_building_pixels": 3}
    )
    assert analysis_res.status_code == 200

    # 3. Wait/check job status until completed
    import time
    for _ in range(30):
        status_res = client.get(f"/api/jobs/{job_id}")
        assert status_res.status_code == 200
        st = status_res.json()
        if st["status"] == "completed":
            break
        time.sleep(0.5)

    assert st["status"] == "completed"
    assert st["progress"] == 100

    # 4. Fetch results
    results_res = client.get(f"/api/jobs/{job_id}/results")
    assert results_res.status_code == 200
    results = results_res.json()
    assert results["building_count"] > 0
    assert results["total_built_area_sqm"] > 0
    assert results["threshold_used"] == 0.75

    # 5. Fetch GeoJSON buildings
    buildings_res = client.get(f"/api/jobs/{job_id}/buildings")
    assert buildings_res.status_code == 200
    geojson = buildings_res.json()
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == results["building_count"]

    first_building = geojson["features"][0]
    b_id = first_building["properties"]["building_id"]
    original_count = len(geojson["features"])

    # 6. Test renaming building
    new_name = "North Sector Warehouse B"
    rename_res = client.patch(
        f"/api/jobs/{job_id}/buildings/{b_id}",
        json={"name": new_name}
    )
    assert rename_res.status_code == 200
    assert rename_res.json()["updated_building"]["properties"]["name"] == new_name
    assert rename_res.json()["updated_building"]["properties"]["status"] == "user_edited"

    # Verify rename persisted
    verify_res = client.get(f"/api/jobs/{job_id}/buildings")
    updated_feature = next(
        f for f in verify_res.json()["features"]
        if f["properties"]["building_id"] == b_id
    )
    assert updated_feature["properties"]["name"] == new_name
    assert updated_feature["properties"]["status"] == "user_edited"

    # 7. Test deleting building
    delete_res = client.delete(f"/api/jobs/{job_id}/buildings/{b_id}")
    assert delete_res.status_code == 200
    assert delete_res.json()["status"] == "success"
    assert delete_res.json()["remaining_count"] == original_count - 1

    # Verify deletion in GeoJSON
    after_del_res = client.get(f"/api/jobs/{job_id}/buildings")
    remaining_ids = [f["properties"]["building_id"] for f in after_del_res.json()["features"]]
    assert b_id not in remaining_ids

    # 8. Test GeoJSON export endpoint
    export_res = client.get(f"/api/jobs/{job_id}/export/geojson")
    assert export_res.status_code == 200
    assert "geo+json" in export_res.headers.get("content-type", "")

    # 9. Test downloading artifacts
    for artifact in ["mask_tif", "probability_tif", "overlay_png", "summary"]:
        dl_res = client.get(f"/api/jobs/{job_id}/download/{artifact}")
        assert dl_res.status_code == 200

    # 10. Test dynamic threshold adjustment without re-inference
    thresh_res = client.post(
        f"/api/jobs/{job_id}/threshold",
        json={"threshold": 0.60, "min_building_pixels": 3}
    )
    assert thresh_res.status_code == 200
    thresh_data = thresh_res.json()
    assert thresh_data["threshold"] == 0.60
    assert "geojson" in thresh_data
    assert thresh_data["building_count"] > 0

    # 11. Test ground truth retrieval for SpaceNet sample
    gt_res = client.get(f"/api/jobs/{job_id}/groundtruth")
    assert gt_res.status_code == 200
    gt_data = gt_res.json()
    assert "features" in gt_data
    assert gt_data["available"] is True

    # 12. Test single building export
    remaining_feat = after_del_res.json()["features"][0]
    rem_id = remaining_feat["properties"]["building_id"]
    single_res = client.get(f"/api/jobs/{job_id}/buildings/{rem_id}/export")
    assert single_res.status_code == 200
    assert single_res.json()["properties"]["building_id"] == rem_id
