from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from rasterio.features import shapes
from rasterio.transform import rowcol
from rasterio.warp import transform_geom
from shapely.geometry import shape, mapping, Polygon, MultiPolygon


def calculate_polygon_metric_area(geom_wgs84: Dict[str, Any], center_lat: float) -> float:
    """
    Calculate approximate surface area in square meters for a WGS84 polygon.
    Uses latitude-adjusted metric scale factors.
    """
    try:
        poly = shape(geom_wgs84)
        if not poly.is_valid or poly.is_empty:
            poly = poly.buffer(0)
            if poly.is_empty:
                return 0.0

        # Degree to meter factors at this latitude
        lat_rad = np.radians(center_lat)
        meters_per_deg_lat = 111132.954 - 559.822 * np.cos(2 * lat_rad) + 1.175 * np.cos(4 * lat_rad)
        meters_per_deg_lon = (np.pi / 180.0) * 6378137.0 * np.cos(lat_rad)

        scale_factor = meters_per_deg_lat * meters_per_deg_lon
        return float(abs(poly.area * scale_factor))
    except Exception:
        return 0.0


def extract_building_polygons(
    prediction: np.ndarray,
    probabilities: np.ndarray,
    transform: Any,
    source_crs: Any,
    output_geojson_path: Path,
    source_name: str = "satellite_image",
    min_area_pixels: int = 3,
) -> Tuple[Dict[str, Any], int, float]:
    """
    Extract building footprint polygons from prediction mask, reproject to WGS84,
    calculate metric area, and save as GeoJSON FeatureCollection.
    """
    features: List[Dict[str, Any]] = []
    building_count = 0
    total_area_sqm = 0.0

    # Source pixel area in map units
    pixel_area_map_units = abs(transform.a * transform.e - transform.b * transform.d)
    if pixel_area_map_units <= 0:
        pixel_area_map_units = 1.0

    # Extract polygon shapes from binary mask (connected components = 4)
    raw_shapes = shapes(
        prediction,
        mask=(prediction == 1),
        transform=transform,
        connectivity=4,
    )

    for geom_src, val in raw_shapes:
        if int(val) != 1:
            continue

        try:
            poly_src = shape(geom_src)
            if not poly_src.is_valid:
                poly_src = poly_src.buffer(0)
            if poly_src.is_empty:
                continue

            area_pixels = int(round(poly_src.area / pixel_area_map_units))
            if area_pixels < min_area_pixels:
                continue

            # Reproject to WGS84 (EPSG:4326)
            if str(source_crs).upper() != "EPSG:4326":
                geom_wgs84 = transform_geom(source_crs, "EPSG:4326", geom_src, precision=7)
            else:
                geom_wgs84 = geom_src

            poly_wgs84 = shape(geom_wgs84)
            centroid = poly_wgs84.centroid
            center_lat = centroid.y
            center_lon = centroid.x

            # Calculate metric area
            area_sqm = calculate_polygon_metric_area(geom_wgs84, center_lat)

            # Calculate confidence inside polygon bounding box
            min_x, min_y, max_x, max_y = poly_src.bounds
            row1, col1 = rowcol(transform, min_x, max_y)
            row2, col2 = rowcol(transform, max_x, min_y)
            r0 = max(0, int(min(row1, row2)))
            r1 = min(prediction.shape[0], int(max(row1, row2)) + 1)
            c0 = max(0, int(min(col1, col2)))
            c1 = min(prediction.shape[1], int(max(col1, col2)) + 1)

            sub_prob = probabilities[r0:r1, c0:c1]
            sub_mask = prediction[r0:r1, c0:c1] == 1
            if sub_mask.any():
                confidence = float(np.mean(sub_prob[sub_mask]))
            else:
                confidence = float(np.mean(sub_prob)) if sub_prob.size > 0 else 0.85

            building_count += 1
            total_area_sqm += area_sqm

            building_id = f"building_{building_count:05d}"
            feature = {
                "type": "Feature",
                "id": building_id,
                "properties": {
                    "building_id": building_id,
                    "name": f"Building {building_count}",
                    "area_pixels": area_pixels,
                    "area_sqm": round(area_sqm, 2),
                    "confidence": round(confidence, 4),
                    "status": "ai_detected",
                    "source_image": source_name,
                    "centroid": [round(center_lat, 6), round(center_lon, 6)],
                },
                "geometry": geom_wgs84,
            }
            features.append(feature)

        except Exception as exc:
            print(f"Skipped polygon due to geometry conversion error: {exc}")
            continue

    geojson_collection = {
        "type": "FeatureCollection",
        "name": output_geojson_path.stem,
        "crs": {
            "type": "name",
            "properties": {"name": "urn:ogc:def:crs:OGC:1.3:CRS84"},
        },
        "features": features,
    }

    output_geojson_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_geojson_path, "w", encoding="utf-8") as f:
        json.dump(geojson_collection, f, indent=2)

    return geojson_collection, building_count, total_area_sqm


def load_geojson(geojson_path: Path) -> Dict[str, Any]:
    if not geojson_path.exists():
        raise FileNotFoundError(f"GeoJSON not found: {geojson_path}")
    with open(geojson_path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_geojson(geojson_path: Path, data: Dict[str, Any]) -> None:
    geojson_path.parent.mkdir(parents=True, exist_ok=True)
    with open(geojson_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def update_building_property(
    geojson_path: Path,
    building_id: str,
    name: Optional[str] = None,
    notes: Optional[str] = None,
    building_type: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Update properties of an existing building feature."""
    data = load_geojson(geojson_path)
    updated_feature = None

    for feat in data.get("features", []):
        props = feat.get("properties", {})
        if props.get("building_id") == building_id or feat.get("id") == building_id:
            if name is not None:
                props["name"] = name.strip()
            if notes is not None:
                props["notes"] = notes.strip()
            if building_type is not None:
                props["building_type"] = building_type.strip()
            props["status"] = "user_edited"
            updated_feature = feat
            break

    if updated_feature:
        save_geojson(geojson_path, data)

    return updated_feature


def delete_building_feature(
    geojson_path: Path,
    building_id: str
) -> bool:
    """Delete a building polygon from the GeoJSON."""
    data = load_geojson(geojson_path)
    original_len = len(data.get("features", []))

    data["features"] = [
        feat for feat in data.get("features", [])
        if feat.get("properties", {}).get("building_id") != building_id and feat.get("id") != building_id
    ]

    if len(data["features"]) < original_len:
        save_geojson(geojson_path, data)
        return True
    return False
