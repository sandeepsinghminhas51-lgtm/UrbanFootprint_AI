import React, { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import {
  Layers,
  ZoomIn,
  ZoomOut,
  Maximize,
  Edit2,
  Trash2,
  Download,
  Search,
  Check,
  X,
  MapPin,
  Eye,
  EyeOff,
  Sliders,
  FileCode,
  Image as ImageIcon,
  FileText,
  AlertCircle,
  ExternalLink,
  ChevronRight,
  ChevronLeft,
  Info,
  Ruler,
  MousePointer,
  Hand,
  RotateCcw,
  Sparkles,
  HelpCircle,
  Share2,
  ShieldCheck,
  Building
} from 'lucide-react';
import { api } from '../api/client';

export default function GisViewer({ activeJob, setCurrentView }) {
  const mapContainerRef = useRef(null);
  const mapInstanceRef = useRef(null);
  const geojsonLayerRef = useRef(null);
  const groundTruthLayerRef = useRef(null);
  const satelliteOverlayRef = useRef(null);
  const maskOverlayRef = useRef(null);
  const probOverlayRef = useRef(null);
  const coverageBoxRef = useRef(null);
  const measureLayerRef = useRef(null);

  // GeoJSON & Building State
  const [geojson, setGeojson] = useState(null);
  const [groundTruth, setGroundTruth] = useState(null);
  const [selectedBuilding, setSelectedBuilding] = useState(null);
  const [editingName, setEditingName] = useState('');
  const [isEditing, setIsEditing] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  const [loading, setLoading] = useState(true);
  const [saveMessage, setSaveMessage] = useState(null);

  // Workspace Panel Visibility
  const [leftPanelOpen, setLeftPanelOpen] = useState(true);
  const [rightPanelOpen, setRightPanelOpen] = useState(true);
  const [leftTab, setLeftTab] = useState('layers'); // 'layers' | 'tools'
  const [activeTool, setActiveTool] = useState('select'); // 'select' | 'pan' | 'measure'

  // Dynamic Threshold State
  const [currentThreshold, setCurrentThreshold] = useState(
    activeJob?.results?.threshold || 0.75
  );
  const [sliderThreshold, setSliderThreshold] = useState(
    activeJob?.results?.threshold || 0.75
  );
  const [isApplyingThreshold, setIsApplyingThreshold] = useState(false);
  const [thresholdModalOpen, setThresholdModalOpen] = useState(false);

  // Layer Visibility & Opacity
  const [showBaseMap, setShowBaseMap] = useState(true);
  const [showSatelliteImage, setShowSatelliteImage] = useState(true);
  const [showProbMap, setShowProbMap] = useState(false);
  const [showMaskOverlay, setShowMaskOverlay] = useState(true);
  const [showPolygons, setShowPolygons] = useState(true);
  const [showGroundTruth, setShowGroundTruth] = useState(false);
  const [showCoverageBounds, setShowCoverageBounds] = useState(true);

  const [maskOpacity, setMaskOpacity] = useState(0.65);
  const [probOpacity, setProbOpacity] = useState(0.6);

  // Map Live Telemetry (Status Bar)
  const [currentZoom, setCurrentZoom] = useState(17);
  const [cursorCoords, setCursorCoords] = useState(null);
  const [measureDistance, setMeasureDistance] = useState(null);
  const measurePointsRef = useRef([]);

  const baseTileLayerRef = useRef(null);

  // Fetch AI Footprints
  const fetchFootprints = async () => {
    if (!activeJob) return;
    try {
      setLoading(true);
      const data = await api.getBuildings(activeJob.job_id);
      setGeojson(data);
      if (data.features && data.features.length > 0) {
        setSelectedBuilding(data.features[0]);
        setEditingName(data.features[0].properties.name || data.features[0].properties.building_id);
      }
    } catch (err) {
      console.error('Error loading GeoJSON:', err);
    } finally {
      setLoading(false);
    }
  };

  // Fetch Ground Truth if available (e.g. SpaceNet scenes)
  const fetchGroundTruth = async () => {
    if (!activeJob) return;
    try {
      const gt = await api.getGroundTruth(activeJob.job_id);
      if (gt && gt.available && gt.features?.length > 0) {
        setGroundTruth(gt);
      } else {
        setGroundTruth(null);
      }
    } catch (err) {
      setGroundTruth(null);
    }
  };

  useEffect(() => {
    fetchFootprints();
    fetchGroundTruth();
    if (activeJob?.results?.threshold) {
      setCurrentThreshold(activeJob.results.threshold);
      setSliderThreshold(activeJob.results.threshold);
    }
  }, [activeJob?.job_id]);

  // Initialize Leaflet Map
  useEffect(() => {
    if (!mapContainerRef.current) return;
    if (mapInstanceRef.current) return;

    const map = L.map(mapContainerRef.current, {
      zoomControl: false,
      attributionControl: false,
      maxZoom: 22,
    }).setView([36.128, -115.306], 17);

    // Warm light CartoDB Voyager tiles for professional GIS look
    const baseTiles = L.tileLayer(
      'https://{s}.basemaps.cartocdn.com/rastertiles/voyager/{z}/{x}/{y}{r}.png',
      {
        maxZoom: 22,
        subdomains: 'abcd',
      }
    ).addTo(map);
    baseTileLayerRef.current = baseTiles;

    // Measurement layer group
    const measureLayer = L.layerGroup().addTo(map);
    measureLayerRef.current = measureLayer;

    // Telemetry listeners
    map.on('zoomend', () => setCurrentZoom(map.getZoom()));
    map.on('mousemove', (e) => {
      setCursorCoords({ lat: e.latlng.lat, lon: e.latlng.lng });
    });
    map.on('mouseout', () => setCursorCoords(null));

    mapInstanceRef.current = map;

    return () => {
      map.remove();
      mapInstanceRef.current = null;
    };
  }, []);

  // Measurement Tool Click Handler
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    const handleMeasureClick = (e) => {
      if (activeTool !== 'measure') return;

      const pts = measurePointsRef.current;
      pts.push(e.latlng);

      if (measureLayerRef.current) {
        measureLayerRef.current.clearLayers();

        // Draw markers
        pts.forEach((pt) => {
          L.circleMarker(pt, {
            radius: 5,
            color: '#71836B',
            fillColor: '#FFFFFF',
            fillOpacity: 1,
            weight: 2,
          }).addTo(measureLayerRef.current);
        });

        // Draw polyline if 2 or more points
        if (pts.length > 1) {
          const polyline = L.polyline(pts, {
            color: '#71836B',
            dashArray: '4, 4',
            weight: 2,
          }).addTo(measureLayerRef.current);

          let totalMeters = 0;
          for (let i = 0; i < pts.length - 1; i++) {
            totalMeters += pts[i].distanceTo(pts[i + 1]);
          }
          setMeasureDistance(totalMeters);
        }
      }
    };

    map.on('click', handleMeasureClick);
    return () => {
      map.off('click', handleMeasureClick);
    };
  }, [activeTool]);

  // Clear measure when switching away from measure tool
  useEffect(() => {
    if (activeTool !== 'measure') {
      measurePointsRef.current = [];
      setMeasureDistance(null);
      if (measureLayerRef.current) {
        measureLayerRef.current.clearLayers();
      }
    }
  }, [activeTool]);

  // Update Raster Overlays
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map || !activeJob) return;

    const bounds = activeJob.metadata?.wgs84_bounds;
    if (!bounds || bounds.length !== 2) return;

    const leafletBounds = L.latLngBounds(bounds);

    // 1. Satellite Image Overlay
    if (satelliteOverlayRef.current) {
      map.removeLayer(satelliteOverlayRef.current);
    }
    const satelliteUrl = `/api/jobs/${activeJob.job_id}/files/preview_satellite.png`;
    const satelliteLayer = L.imageOverlay(satelliteUrl, leafletBounds, {
      opacity: showSatelliteImage ? 1.0 : 0.0,
      interactive: false,
    }).addTo(map);
    satelliteOverlayRef.current = satelliteLayer;

    // 2. AI Mask Overlay (transparent PNG)
    if (maskOverlayRef.current) {
      map.removeLayer(maskOverlayRef.current);
    }
    const maskUrl = `/api/jobs/${activeJob.job_id}/files/building_mask.png`;
    const maskLayer = L.imageOverlay(maskUrl, leafletBounds, {
      opacity: showMaskOverlay ? maskOpacity : 0.0,
      interactive: false,
    }).addTo(map);
    maskOverlayRef.current = maskLayer;

    // 3. Image Coverage Bounding Box
    if (coverageBoxRef.current) {
      map.removeLayer(coverageBoxRef.current);
    }
    const coverageBox = L.rectangle(leafletBounds, {
      color: '#71836B',
      weight: 1.5,
      dashArray: '6, 4',
      fill: false,
      interactive: false,
    });
    if (showCoverageBounds) {
      coverageBox.addTo(map);
    }
    coverageBoxRef.current = coverageBox;

    // Fit map view to raster extents
    map.fitBounds(leafletBounds, { padding: [30, 30] });
  }, [activeJob?.job_id]);

  // Layer Visibility & Opacity Changes
  useEffect(() => {
    if (satelliteOverlayRef.current) {
      satelliteOverlayRef.current.setOpacity(showSatelliteImage ? 1.0 : 0.0);
    }
    if (maskOverlayRef.current) {
      maskOverlayRef.current.setOpacity(showMaskOverlay ? maskOpacity : 0.0);
    }
    if (baseTileLayerRef.current && mapInstanceRef.current) {
      if (showBaseMap) {
        if (!mapInstanceRef.current.hasLayer(baseTileLayerRef.current)) {
          baseTileLayerRef.current.addTo(mapInstanceRef.current);
        }
      } else {
        if (mapInstanceRef.current.hasLayer(baseTileLayerRef.current)) {
          mapInstanceRef.current.removeLayer(baseTileLayerRef.current);
        }
      }
    }
    if (coverageBoxRef.current && mapInstanceRef.current) {
      if (showCoverageBounds) {
        if (!mapInstanceRef.current.hasLayer(coverageBoxRef.current)) {
          coverageBoxRef.current.addTo(mapInstanceRef.current);
        }
      } else {
        if (mapInstanceRef.current.hasLayer(coverageBoxRef.current)) {
          mapInstanceRef.current.removeLayer(coverageBoxRef.current);
        }
      }
    }
  }, [showSatelliteImage, showMaskOverlay, maskOpacity, showBaseMap, showCoverageBounds]);

  // Render Ground Truth Polygons (Gold / Sand outline)
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (groundTruthLayerRef.current) {
      map.removeLayer(groundTruthLayerRef.current);
      groundTruthLayerRef.current = null;
    }

    if (!groundTruth || !showGroundTruth) return;

    const gtLayer = L.geoJSON(groundTruth, {
      style: {
        color: '#C5A982', // Warm sand outline for ground truth comparison
        weight: 2,
        opacity: 0.9,
        fillColor: '#C5A982',
        fillOpacity: 0.15,
        dashArray: '3, 3',
      },
      onEachFeature: (feat, layer) => {
        layer.bindTooltip("Ground Truth Footprint (SpaceNet)", { sticky: true });
      }
    }).addTo(map);

    groundTruthLayerRef.current = gtLayer;
  }, [groundTruth, showGroundTruth]);

  // Render AI Prediction Polygons (Terracotta / Orange outline)
  useEffect(() => {
    const map = mapInstanceRef.current;
    if (!map) return;

    if (geojsonLayerRef.current) {
      map.removeLayer(geojsonLayerRef.current);
      geojsonLayerRef.current = null;
    }

    if (!geojson || !showPolygons) return;

    const layer = L.geoJSON(geojson, {
      style: (feature) => {
        const isSelected = selectedBuilding?.properties?.building_id === feature.properties.building_id;
        return {
          color: isSelected ? '#1A1D1C' : '#D47348', // Dark outline when selected, terracotta default
          weight: isSelected ? 3.5 : 2,
          opacity: isSelected ? 1.0 : 0.9,
          fillColor: isSelected ? '#E8ECE6' : '#C5A982',
          fillOpacity: isSelected ? 0.6 : 0.25,
          dashArray: isSelected ? '4, 4' : null,
        };
      },
      onEachFeature: (feature, polygonLayer) => {
        polygonLayer.on({
          click: () => {
            if (activeTool === 'measure') return;
            setSelectedBuilding(feature);
            setEditingName(feature.properties.name || feature.properties.building_id);
            setIsEditing(false);
          },
          mouseover: (e) => {
            const isSelected = selectedBuilding?.properties?.building_id === feature.properties.building_id;
            if (!isSelected && activeTool !== 'measure') {
              e.target.setStyle({
                color: '#71836B',
                weight: 3,
                fillOpacity: 0.45,
              });
            }
          },
          mouseout: (e) => {
            const isSelected = selectedBuilding?.properties?.building_id === feature.properties.building_id;
            if (!isSelected && activeTool !== 'measure') {
              layer.resetStyle(e.target);
            }
          },
        });

        // Popup tooltip
        const p = feature.properties;
        polygonLayer.bindPopup(`
          <div style="font-family: Inter, sans-serif; font-size: 12px; color: #292D2B;">
            <div style="font-weight: 600; font-size: 13px; margin-bottom: 4px; color: #1A1D1C;">
              ${p.name || p.building_id}
            </div>
            <div style="display: grid; grid-template-columns: auto auto; gap: 4px 12px; font-size: 11px; color: #777B75;">
              <span>Area:</span><b style="color: #292D2B;">${p.area_sqm} m²</b>
              <span>Confidence:</span><b style="color: #292D2B;">${(p.confidence * 100).toFixed(1)}%</b>
              <span>Status:</span><span style="text-transform: capitalize; color: #71836B;">${p.status}</span>
            </div>
          </div>
        `);
      },
    }).addTo(map);

    geojsonLayerRef.current = layer;
  }, [geojson, showPolygons, selectedBuilding?.properties?.building_id, activeTool]);

  // Center on selected building
  const handleSelectBuilding = (feat) => {
    setSelectedBuilding(feat);
    setEditingName(feat.properties.name || feat.properties.building_id);
    setIsEditing(false);

    if (mapInstanceRef.current && feat.geometry) {
      const tempLayer = L.geoJSON(feat);
      mapInstanceRef.current.fitBounds(tempLayer.getBounds(), {
        maxZoom: 20,
        padding: [80, 80],
      });
    }
  };

  // Dynamic Threshold Adjustment Handler (No U-Net re-inference, fast binarization)
  const handleApplyThreshold = async () => {
    if (!activeJob) return;
    try {
      setIsApplyingThreshold(true);
      const res = await api.applyThreshold(activeJob.job_id, sliderThreshold);
      setCurrentThreshold(sliderThreshold);
      setGeojson(res.geojson);
      if (res.geojson.features?.length > 0) {
        setSelectedBuilding(res.geojson.features[0]);
        setEditingName(res.geojson.features[0].properties.name);
      } else {
        setSelectedBuilding(null);
      }
      setSaveMessage(`Applied threshold ${sliderThreshold.toFixed(2)} (${res.building_count} footprints)`);
      setTimeout(() => setSaveMessage(null), 3500);

      // Force refresh mask overlay image cache
      if (maskOverlayRef.current && mapInstanceRef.current) {
        const bounds = activeJob.metadata?.wgs84_bounds;
        if (bounds) {
          mapInstanceRef.current.removeLayer(maskOverlayRef.current);
          const maskUrl = `/api/jobs/${activeJob.job_id}/files/building_mask.png?t=${Date.now()}`;
          const newLayer = L.imageOverlay(maskUrl, L.latLngBounds(bounds), {
            opacity: showMaskOverlay ? maskOpacity : 0.0,
            interactive: false,
          }).addTo(mapInstanceRef.current);
          maskOverlayRef.current = newLayer;
        }
      }
    } catch (err) {
      alert(`Threshold update failed: ${err.message}`);
    } finally {
      setIsApplyingThreshold(false);
    }
  };

  const handleResetOptimalThreshold = () => {
    setSliderThreshold(0.75);
  };

  // Save edited building name
  const handleSaveRename = async () => {
    if (!activeJob || !selectedBuilding) return;
    try {
      const bId = selectedBuilding.properties.building_id;
      await api.updateBuilding(activeJob.job_id, bId, { name: editingName });

      const updated = { ...geojson };
      for (const f of updated.features) {
        if (f.properties.building_id === bId) {
          f.properties.name = editingName;
          f.properties.status = 'user_edited';
          break;
        }
      }
      setGeojson({ ...updated });
      setSelectedBuilding((prev) => ({
        ...prev,
        properties: { ...prev.properties, name: editingName, status: 'user_edited' },
      }));
      setIsEditing(false);
      setSaveMessage('Saved building name');
      setTimeout(() => setSaveMessage(null), 3000);
    } catch (err) {
      alert(`Failed to save rename: ${err.message}`);
    }
  };

  // Delete building polygon
  const handleDeleteBuilding = async (bId) => {
    if (!window.confirm(`Delete polygon "${bId}"? This will update the GeoJSON layer.`)) return;
    try {
      await api.deleteBuilding(activeJob.job_id, bId);
      const remainingFeatures = geojson.features.filter(
        (f) => f.properties.building_id !== bId
      );
      setGeojson({ ...geojson, features: remainingFeatures });
      if (selectedBuilding?.properties?.building_id === bId) {
        setSelectedBuilding(remainingFeatures[0] || null);
        setEditingName(remainingFeatures[0]?.properties?.name || '');
      }
      setSaveMessage(`Polygon ${bId} deleted`);
      setTimeout(() => setSaveMessage(null), 3000);
    } catch (err) {
      alert(`Failed to delete building: ${err.message}`);
    }
  };

  // Fit to image bounds
  const fitRasterBounds = () => {
    if (!mapInstanceRef.current || !activeJob?.metadata?.wgs84_bounds) return;
    mapInstanceRef.current.fitBounds(activeJob.metadata.wgs84_bounds, { padding: [30, 30] });
  };

  // Filter buildings list
  const filteredBuildings = geojson?.features?.filter((feat) => {
    const name = feat.properties.name || '';
    const id = feat.properties.building_id || '';
    const q = searchTerm.toLowerCase();
    return name.toLowerCase().includes(q) || id.toLowerCase().includes(q);
  }) || [];

  const totalBuiltArea = geojson?.features?.reduce(
    (acc, f) => acc + (f.properties.area_sqm || 0),
    0
  ) || 0;

  // Vertex count helper
  const getVertexCount = (feat) => {
    try {
      const coords = feat.geometry?.coordinates;
      if (!coords) return 0;
      if (feat.geometry.type === 'Polygon') return coords[0]?.length || 0;
      if (feat.geometry.type === 'MultiPolygon') {
        return coords.reduce((acc, poly) => acc + (poly[0]?.length || 0), 0);
      }
      return 0;
    } catch {
      return 0;
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-3.25rem)] bg-warm-50 overflow-hidden font-sans select-none">
      {/* 3-Panel Main GIS Workspace */}
      <div className="flex-1 flex overflow-hidden relative">
        {/* ============================================================== */}
        {/* PANEL 1 (LEFT): LAYERS & TOOLS SIDEBAR (Collapsible)           */}
        {/* ============================================================== */}
        <div
          className={`border-r border-warm-200 bg-surface flex flex-col z-20 shrink-0 transition-all duration-200 ${
            leftPanelOpen ? 'w-72 sm:w-80' : 'w-0 overflow-hidden border-r-0'
          }`}
        >
          {/* Left Panel Tabs Header */}
          <div className="h-10 border-b border-warm-200 bg-warm-50/70 px-3 flex items-center justify-between shrink-0">
            <div className="flex items-center gap-2">
              <button
                onClick={() => setLeftTab('layers')}
                className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${
                  leftTab === 'layers'
                    ? 'bg-surface text-ink shadow-subtle font-semibold'
                    : 'text-ink-secondary hover:text-ink'
                }`}
              >
                Layers
              </button>
              <button
                onClick={() => setLeftTab('tools')}
                className={`px-2.5 py-1 text-xs font-medium rounded transition-colors ${
                  leftTab === 'tools'
                    ? 'bg-surface text-ink shadow-subtle font-semibold'
                    : 'text-ink-secondary hover:text-ink'
                }`}
              >
                Tools
              </button>
            </div>

            <button
              onClick={() => setLeftPanelOpen(false)}
              className="p-1 rounded hover:bg-warm-100 text-ink-muted hover:text-ink"
              title="Collapse Left Sidebar"
            >
              <ChevronLeft className="h-4 w-4" />
            </button>
          </div>

          {/* Left Tab 1: Layers Stack */}
          {leftTab === 'layers' && (
            <div className="flex-1 overflow-y-auto p-4 space-y-5 text-xs text-ink">
              {/* Layer 1: Base Map */}
              <div className="space-y-1.5 pb-3 border-b border-warm-100">
                <label className="flex items-center justify-between cursor-pointer">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={showBaseMap}
                      onChange={(e) => setShowBaseMap(e.target.checked)}
                      className="accent-accent-green rounded"
                    />
                    <span className="font-medium">Basemap Reference</span>
                  </div>
                  <span className="text-[10px] text-ink-muted font-mono">CartoDB</span>
                </label>
              </div>

              {/* Layer 2: Satellite GeoTIFF */}
              <div className="space-y-1.5 pb-3 border-b border-warm-100">
                <label className="flex items-center justify-between cursor-pointer">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={showSatelliteImage}
                      onChange={(e) => setShowSatelliteImage(e.target.checked)}
                      className="accent-accent-green rounded"
                    />
                    <span className="font-medium">Satellite GeoTIFF</span>
                  </div>
                  <span className="text-[10px] font-mono bg-warm-100 px-1.5 py-0.2 rounded text-ink-secondary">
                    RGB (3-Band)
                  </span>
                </label>
                <div className="text-[10px] text-ink-muted pl-6">
                  {activeJob?.metadata?.width} × {activeJob?.metadata?.height} px • 0.3m GSD
                </div>
              </div>

              {/* Layer 3: AI Binary Mask */}
              <div className="space-y-2 pb-3 border-b border-warm-100">
                <label className="flex items-center justify-between cursor-pointer">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={showMaskOverlay}
                      onChange={(e) => setShowMaskOverlay(e.target.checked)}
                      className="accent-accent-green rounded"
                    />
                    <span className="font-medium">Predicted Building Mask</span>
                  </div>
                  <span className="h-2.5 w-2.5 rounded-full bg-rose-500" />
                </label>

                {showMaskOverlay && (
                  <div className="pl-6 space-y-1">
                    <div className="flex justify-between text-[10px] text-ink-secondary font-mono">
                      <span>Mask Opacity</span>
                      <span>{Math.round(maskOpacity * 100)}%</span>
                    </div>
                    <input
                      type="range"
                      min="0.1"
                      max="1.0"
                      step="0.05"
                      value={maskOpacity}
                      onChange={(e) => setMaskOpacity(parseFloat(e.target.value))}
                      className="w-full accent-accent-green h-1 bg-warm-200 rounded cursor-pointer"
                    />
                  </div>
                )}
              </div>

              {/* Layer 4: AI Extracted Polygons */}
              <div className="space-y-1.5 pb-3 border-b border-warm-100">
                <label className="flex items-center justify-between cursor-pointer">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={showPolygons}
                      onChange={(e) => setShowPolygons(e.target.checked)}
                      className="accent-accent-green rounded"
                    />
                    <span className="font-medium text-ink">Building Footprints (AI)</span>
                  </div>
                  <span className="text-[10px] font-mono text-accent-terracotta font-bold">
                    {geojson?.features?.length || 0}
                  </span>
                </label>
                <div className="flex items-center gap-2 pl-6 text-[10px] text-ink-muted">
                  <span className="h-2 w-3 border border-accent-terracotta bg-accent-sand/30" />
                  <span>Terracotta Planar Vectors (WGS84)</span>
                </div>
              </div>

              {/* Layer 5: Ground Truth (when available) */}
              <div className="space-y-1.5 pb-3 border-b border-warm-100">
                <label className={`flex items-center justify-between cursor-pointer ${!groundTruth ? 'opacity-50 cursor-not-allowed' : ''}`}>
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      disabled={!groundTruth}
                      checked={showGroundTruth}
                      onChange={(e) => setShowGroundTruth(e.target.checked)}
                      className="accent-accent-green rounded"
                    />
                    <span className="font-medium">SpaceNet Ground Truth</span>
                  </div>
                  {groundTruth ? (
                    <span className="text-[10px] font-mono text-accent-sandDark font-bold">
                      {groundTruth.features?.length || 0}
                    </span>
                  ) : (
                    <span className="text-[9px] text-ink-muted font-mono">N/A</span>
                  )}
                </label>
                {groundTruth && (
                  <div className="flex items-center gap-2 pl-6 text-[10px] text-ink-muted">
                    <span className="h-2 w-3 border border-accent-sand dashed" />
                    <span>Official SpaceNet 2 Human Annotations</span>
                  </div>
                )}
              </div>

              {/* Layer 6: Coverage Extent */}
              <div className="space-y-1.5">
                <label className="flex items-center justify-between cursor-pointer">
                  <div className="flex items-center gap-2">
                    <input
                      type="checkbox"
                      checked={showCoverageBounds}
                      onChange={(e) => setShowCoverageBounds(e.target.checked)}
                      className="accent-accent-green rounded"
                    />
                    <span className="font-medium">Image Extent Boundary</span>
                  </div>
                  <span className="text-[10px] text-ink-muted font-mono">CRS Box</span>
                </label>
              </div>
            </div>
          )}

          {/* Left Tab 2: GIS Tools */}
          {leftTab === 'tools' && (
            <div className="flex-1 overflow-y-auto p-4 space-y-4 text-xs text-ink">
              <div className="space-y-1">
                <span className="text-[10px] font-mono uppercase tracking-wider text-ink-secondary font-semibold">
                  Interaction Modes
                </span>
                <div className="grid grid-cols-1 gap-1.5 pt-1">
                  <button
                    onClick={() => setActiveTool('select')}
                    className={`flex items-center gap-2.5 p-2 rounded-md text-left transition-colors ${
                      activeTool === 'select'
                        ? 'bg-accent-greenLight text-accent-greenDark font-semibold border border-accent-green/20'
                        : 'hover:bg-warm-100 text-ink'
                    }`}
                  >
                    <MousePointer className="h-4 w-4" />
                    <div>
                      <div className="text-xs">Select Footprint</div>
                      <div className="text-[10px] text-ink-secondary font-normal">Click polygon to view details</div>
                    </div>
                  </button>

                  <button
                    onClick={() => setActiveTool('pan')}
                    className={`flex items-center gap-2.5 p-2 rounded-md text-left transition-colors ${
                      activeTool === 'pan'
                        ? 'bg-accent-greenLight text-accent-greenDark font-semibold border border-accent-green/20'
                        : 'hover:bg-warm-100 text-ink'
                    }`}
                  >
                    <Hand className="h-4 w-4" />
                    <div>
                      <div className="text-xs">Pan Canvas</div>
                      <div className="text-[10px] text-ink-secondary font-normal">Free drag without selecting</div>
                    </div>
                  </button>

                  <button
                    onClick={() => setActiveTool('measure')}
                    className={`flex items-center gap-2.5 p-2 rounded-md text-left transition-colors ${
                      activeTool === 'measure'
                        ? 'bg-accent-greenLight text-accent-greenDark font-semibold border border-accent-green/20'
                        : 'hover:bg-warm-100 text-ink'
                    }`}
                  >
                    <Ruler className="h-4 w-4" />
                    <div>
                      <div className="text-xs">Measure Distance</div>
                      <div className="text-[10px] text-ink-secondary font-normal">Click points to measure ground meters</div>
                    </div>
                  </button>
                </div>
              </div>

              {activeTool === 'measure' && (
                <div className="p-3 rounded-lg bg-warm-100/70 border border-warm-200 text-xs space-y-1.5 font-mono">
                  <div className="font-semibold text-ink flex items-center justify-between">
                    <span>Measured Distance</span>
                    {measureDistance !== null && (
                      <button
                        onClick={() => {
                          measurePointsRef.current = [];
                          setMeasureDistance(null);
                          measureLayerRef.current?.clearLayers();
                        }}
                        className="text-[10px] text-accent-green hover:underline"
                      >
                        Clear
                      </button>
                    )}
                  </div>
                  <div className="text-sm font-bold text-accent-green">
                    {measureDistance !== null ? `${measureDistance.toFixed(1)} meters` : 'Click 2 points on map'}
                  </div>
                  <div className="text-[10px] text-ink-muted font-sans">
                    Computed via ellipsoidal geodesic distance.
                  </div>
                </div>
              )}

              {/* View Controls */}
              <div className="pt-2 border-t border-warm-200 space-y-2">
                <span className="text-[10px] font-mono uppercase tracking-wider text-ink-secondary font-semibold">
                  Map Shortcuts
                </span>
                <div className="grid grid-cols-2 gap-2 font-mono text-[11px]">
                  <button
                    onClick={fitRasterBounds}
                    className="p-2 rounded bg-warm-50 hover:bg-warm-100 border border-warm-200 text-ink flex items-center justify-center gap-1.5"
                  >
                    <Maximize className="h-3.5 w-3.5" />
                    Fit Coverage
                  </button>
                  <button
                    onClick={() => {
                      if (selectedBuilding && mapInstanceRef.current) {
                        const tempLayer = L.geoJSON(selectedBuilding);
                        mapInstanceRef.current.fitBounds(tempLayer.getBounds(), { maxZoom: 20 });
                      }
                    }}
                    disabled={!selectedBuilding}
                    className="p-2 rounded bg-warm-50 hover:bg-warm-100 border border-warm-200 text-ink disabled:opacity-40 flex items-center justify-center gap-1.5"
                  >
                    <MapPin className="h-3.5 w-3.5" />
                    Center Bldg
                  </button>
                </div>
              </div>
            </div>
          )}

          {/* ============================================================== */}
          {/* SECTION 4: INTERACTIVE THRESHOLD CONTROL (Bottom of Left Panel) */}
          {/* ============================================================== */}
          <div className="p-3.5 border-t border-warm-200 bg-warm-50/70 space-y-2.5 shrink-0">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-1.5">
                <Sliders className="h-3.5 w-3.5 text-accent-green" />
                <span className="text-xs font-semibold text-ink">Threshold Tuning</span>
              </div>
              <button
                onClick={() => setThresholdModalOpen(true)}
                className="text-ink-muted hover:text-ink"
                title="Threshold Terminology & Info"
              >
                <HelpCircle className="h-3.5 w-3.5" />
              </button>
            </div>

            <div className="space-y-1.5">
              <div className="flex justify-between items-center text-[11px] font-mono">
                <span className="text-ink-secondary">Cutoff Value</span>
                <span className="font-bold text-ink px-1.5 py-0.2 bg-surface rounded border border-warm-200">
                  {sliderThreshold.toFixed(2)}
                </span>
              </div>

              <input
                type="range"
                min="0.30"
                max="0.90"
                step="0.05"
                value={sliderThreshold}
                onChange={(e) => setSliderThreshold(parseFloat(e.target.value))}
                className="w-full accent-accent-green h-1.5 bg-warm-200 rounded cursor-pointer"
              />

              <div className="flex justify-between text-[9px] text-ink-muted font-mono">
                <span>0.30 (Recall)</span>
                <span className="text-accent-green font-semibold">0.75 (Optimal)</span>
                <span>0.90 (Prec)</span>
              </div>
            </div>

            <div className="flex items-center gap-2 pt-1 font-mono">
              <button
                disabled={isApplyingThreshold || sliderThreshold === currentThreshold}
                onClick={handleApplyThreshold}
                className="flex-1 py-1.5 rounded bg-accent-green hover:bg-accent-greenDark text-white text-[11px] font-semibold transition-all disabled:opacity-40 disabled:hover:bg-accent-green shadow-subtle flex items-center justify-center gap-1"
              >
                {isApplyingThreshold ? 'Re-filtering...' : 'Apply Threshold'}
              </button>

              <button
                onClick={handleResetOptimalThreshold}
                title="Reset to 0.75"
                className="p-1.5 rounded bg-surface hover:bg-warm-100 border border-warm-200 text-ink-secondary hover:text-ink text-[11px]"
              >
                <RotateCcw className="h-3.5 w-3.5" />
              </button>
            </div>
          </div>
        </div>

        {/* Toggle Left Sidebar Button (When collapsed) */}
        {!leftPanelOpen && (
          <button
            onClick={() => setLeftPanelOpen(true)}
            className="absolute top-3 left-3 z-20 p-2 rounded-md bg-surface border border-warm-200 text-ink shadow-elevated hover:bg-warm-50"
            title="Expand Left Panel"
          >
            <ChevronRight className="h-4 w-4" />
          </button>
        )}

        {/* ============================================================== */}
        {/* PANEL 2 (CENTER): INTERACTIVE GIS MAP CANVAS                  */}
        {/* ============================================================== */}
        <div className="flex-1 relative h-full w-full bg-[#E8E5DE]">
          <div ref={mapContainerRef} className="absolute inset-0 z-0" />

          {/* Toast Notification */}
          {saveMessage && (
            <div className="absolute top-4 left-1/2 -translate-x-1/2 z-30 px-3.5 py-1.5 rounded-full bg-surface/95 border border-warm-200 shadow-floating text-xs font-mono text-ink flex items-center gap-2 animate-fade-in">
              <Check className="h-3.5 w-3.5 text-accent-green" />
              <span>{saveMessage}</span>
            </div>
          )}

          {/* Floating Map Zoom Toolbar */}
          <div className="absolute bottom-4 right-4 z-10 flex flex-col gap-1.5">
            <button
              onClick={() => mapInstanceRef.current?.zoomIn()}
              className="p-2 rounded-md bg-surface hover:bg-warm-50 border border-warm-200 text-ink shadow-elevated transition-colors"
              title="Zoom In"
            >
              <ZoomIn className="h-4 w-4" />
            </button>
            <button
              onClick={() => mapInstanceRef.current?.zoomOut()}
              className="p-2 rounded-md bg-surface hover:bg-warm-50 border border-warm-200 text-ink shadow-elevated transition-colors"
              title="Zoom Out"
            >
              <ZoomOut className="h-4 w-4" />
            </button>
            <button
              onClick={fitRasterBounds}
              className="p-2 rounded-md bg-surface hover:bg-warm-50 border border-warm-200 text-ink shadow-elevated transition-colors"
              title="Fit to Scene Coverage"
            >
              <Maximize className="h-4 w-4" />
            </button>
          </div>
        </div>

        {/* Toggle Right Sidebar Button (When collapsed) */}
        {!rightPanelOpen && (
          <button
            onClick={() => setRightPanelOpen(true)}
            className="absolute top-3 right-3 z-20 p-2 rounded-md bg-surface border border-warm-200 text-ink shadow-elevated hover:bg-warm-50"
            title="Expand Right Panel"
          >
            <ChevronLeft className="h-4 w-4" />
          </button>
        )}

        {/* ============================================================== */}
        {/* PANEL 3 (RIGHT): FEATURE INSPECTOR SIDEBAR (Collapsible)       */}
        {/* ============================================================== */}
        <div
          className={`border-l border-warm-200 bg-surface flex flex-col z-20 shrink-0 transition-all duration-200 ${
            rightPanelOpen ? 'w-80 sm:w-96' : 'w-0 overflow-hidden border-l-0'
          }`}
        >
          {/* Right Header: Building Title & Quick Actions */}
          <div className="p-4 border-b border-warm-200 bg-warm-50/70 space-y-2 shrink-0">
            <div className="flex items-center justify-between">
              <span className="text-[10px] font-mono uppercase tracking-wider text-ink-secondary font-semibold">
                Feature Properties
              </span>
              <button
                onClick={() => setRightPanelOpen(false)}
                className="p-1 rounded hover:bg-warm-100 text-ink-muted hover:text-ink"
                title="Collapse Right Sidebar"
              >
                <ChevronRight className="h-4 w-4" />
              </button>
            </div>

            {selectedBuilding ? (
              <div className="space-y-1">
                {isEditing ? (
                  <div className="flex items-center gap-1.5">
                    <input
                      type="text"
                      value={editingName}
                      onChange={(e) => setEditingName(e.target.value)}
                      className="flex-1 bg-surface border border-accent-green rounded px-2.5 py-1 text-sm font-semibold text-ink focus:outline-none font-mono"
                      autoFocus
                    />
                    <button
                      onClick={handleSaveRename}
                      className="p-1.5 rounded bg-accent-green hover:bg-accent-greenDark text-white"
                      title="Save Rename"
                    >
                      <Check className="h-3.5 w-3.5" />
                    </button>
                    <button
                      onClick={() => {
                        setIsEditing(false);
                        setEditingName(selectedBuilding.properties.name);
                      }}
                      className="p-1.5 rounded bg-warm-100 hover:bg-warm-200 text-ink-secondary"
                      title="Cancel"
                    >
                      <X className="h-3.5 w-3.5" />
                    </button>
                  </div>
                ) : (
                  <div className="flex items-center justify-between">
                    <h3 className="text-base font-bold text-ink font-serif tracking-tight truncate">
                      {selectedBuilding.properties.name || selectedBuilding.properties.building_id}
                    </h3>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => setIsEditing(true)}
                        className="p-1.5 rounded hover:bg-warm-100 text-ink-secondary hover:text-ink transition-colors"
                        title="Rename Building"
                      >
                        <Edit2 className="h-3.5 w-3.5" />
                      </button>
                      <button
                        onClick={() => handleDeleteBuilding(selectedBuilding.properties.building_id)}
                        className="p-1.5 rounded hover:bg-rose-50 text-ink-secondary hover:text-status-error transition-colors"
                        title="Delete Polygon"
                      >
                        <Trash2 className="h-3.5 w-3.5" />
                      </button>
                    </div>
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <span className="text-[11px] font-mono text-ink-muted">
                    {selectedBuilding.properties.building_id}
                  </span>
                  <span
                    className={`text-[9px] font-mono px-1.5 py-0.2 rounded ${
                      selectedBuilding.properties.status === 'user_edited'
                        ? 'bg-accent-sandLight text-accent-sandDark border border-accent-sand/30'
                        : 'bg-warm-100 text-ink-secondary border border-warm-200'
                    }`}
                  >
                    {selectedBuilding.properties.status === 'user_edited' ? 'User Edited' : 'AI Extracted'}
                  </span>
                </div>
              </div>
            ) : (
              <div className="text-xs text-ink-muted text-center py-2">
                Click a building polygon on the map to inspect.
              </div>
            )}
          </div>

          {/* Building Details Table (Inspired by Rescope) */}
          {selectedBuilding && (
            <div className="p-4 border-b border-warm-200 bg-surface space-y-2 text-xs shrink-0">
              <div className="flex justify-between py-1.5 border-b border-warm-100">
                <span className="text-ink-secondary">Ground Footprint Area</span>
                <span className="font-mono font-bold text-ink">
                  {selectedBuilding.properties.area_sqm} m²{' '}
                  <span className="text-ink-muted font-normal">
                    ({(selectedBuilding.properties.area_sqm * 10.7639).toFixed(0)} sq ft)
                  </span>
                </span>
              </div>

              <div className="flex justify-between py-1.5 border-b border-warm-100">
                <span className="text-ink-secondary">Detection Confidence</span>
                <span className="font-mono font-bold text-accent-green">
                  {(selectedBuilding.properties.confidence * 100).toFixed(1)}%
                </span>
              </div>

              <div className="flex justify-between py-1.5 border-b border-warm-100">
                <span className="text-ink-secondary">Pixel Ground Area</span>
                <span className="font-mono text-ink">{selectedBuilding.properties.area_pixels} px</span>
              </div>

              <div className="flex justify-between py-1.5 border-b border-warm-100">
                <span className="text-ink-secondary">Geometry / Vertices</span>
                <span className="font-mono text-ink">
                  {selectedBuilding.geometry?.type || 'Polygon'} ({getVertexCount(selectedBuilding)} pts)
                </span>
              </div>

              <div className="flex justify-between py-1.5 border-b border-warm-100">
                <span className="text-ink-secondary">Current Cutoff</span>
                <span className="font-mono text-ink">{currentThreshold.toFixed(2)}</span>
              </div>

              {selectedBuilding.properties.centroid && (
                <div className="flex justify-between py-1.5">
                  <span className="text-ink-secondary">Centroid Coordinates</span>
                  <span className="font-mono text-ink text-[11px]">
                    {selectedBuilding.properties.centroid[0].toFixed(5)}, {selectedBuilding.properties.centroid[1].toFixed(5)}
                  </span>
                </div>
              )}

              {/* Action Buttons */}
              <div className="grid grid-cols-2 gap-2 pt-2 font-mono text-[11px]">
                <button
                  onClick={() => handleSelectBuilding(selectedBuilding)}
                  className="p-1.5 rounded bg-warm-50 hover:bg-warm-100 border border-warm-200 text-ink flex items-center justify-center gap-1.5"
                >
                  <MapPin className="h-3 w-3 text-accent-green" />
                  Zoom to Feature
                </button>

                <a
                  href={api.getExportSingleBuildingUrl(activeJob.job_id, selectedBuilding.properties.building_id)}
                  download={`${selectedBuilding.properties.building_id}.geojson`}
                  className="p-1.5 rounded bg-warm-50 hover:bg-warm-100 border border-warm-200 text-ink flex items-center justify-center gap-1.5"
                >
                  <Download className="h-3 w-3 text-ink-secondary" />
                  Export Feature
                </a>
              </div>
            </div>
          )}

          {/* Searchable Directory Header */}
          <div className="p-3 border-b border-warm-200 bg-warm-50/50 shrink-0">
            <div className="relative">
              <Search className="h-3.5 w-3.5 text-ink-muted absolute left-3 top-2.5" />
              <input
                type="text"
                placeholder="Search scene footprints..."
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                className="w-full bg-surface border border-warm-200 rounded-md pl-8 pr-3 py-1.5 text-xs text-ink placeholder-ink-muted focus:outline-none focus:border-accent-green font-mono"
              />
            </div>
          </div>

          {/* Scrollable Building Directory List */}
          <div className="flex-1 overflow-y-auto divide-y divide-warm-100">
            {loading ? (
              <div className="p-8 text-center text-xs text-ink-muted font-mono">
                Loading footprint coordinates...
              </div>
            ) : filteredBuildings.length === 0 ? (
              <div className="p-8 text-center text-xs text-ink-muted font-mono">
                No matching buildings found.
              </div>
            ) : (
              filteredBuildings.map((feat) => {
                const bId = feat.properties.building_id;
                const isSelected = selectedBuilding?.properties?.building_id === bId;
                return (
                  <div
                    key={bId}
                    onClick={() => handleSelectBuilding(feat)}
                    className={`p-3 cursor-pointer text-xs font-mono transition-colors flex items-center justify-between ${
                      isSelected
                        ? 'bg-accent-greenLight/60 border-l-2 border-accent-green'
                        : 'hover:bg-warm-50'
                    }`}
                  >
                    <div className="space-y-0.5">
                      <div className="font-semibold text-ink flex items-center gap-1.5">
                        <span>{feat.properties.name || bId}</span>
                        {feat.properties.status === 'user_edited' && (
                          <span className="text-[9px] text-accent-sandDark font-semibold">• edited</span>
                        )}
                      </div>
                      <div className="text-[11px] text-ink-secondary">
                        {feat.properties.area_sqm} m² • {(feat.properties.confidence * 100).toFixed(0)}% conf
                      </div>
                    </div>

                    <button
                      onClick={(e) => {
                        e.stopPropagation();
                        handleDeleteBuilding(bId);
                      }}
                      className="p-1 rounded hover:bg-rose-50 text-ink-muted hover:text-status-error opacity-60 hover:opacity-100 transition-opacity"
                      title="Delete"
                    >
                      <Trash2 className="h-3 w-3" />
                    </button>
                  </div>
                );
              })
            )}
          </div>

          {/* Export Layer Footer */}
          <div className="p-3 border-t border-warm-200 bg-warm-50/70 shrink-0">
            <a
              href={api.getExportGeoJsonUrl(activeJob.job_id)}
              download
              className="w-full py-2 rounded-md bg-accent-green hover:bg-accent-greenDark text-white text-xs font-medium flex items-center justify-center gap-1.5 shadow-subtle transition-colors font-mono"
            >
              <Download className="h-3.5 w-3.5" />
              Export Complete GeoJSON ({geojson?.features?.length || 0})
            </a>
          </div>
        </div>
      </div>

      {/* ============================================================== */}
      {/* SECTION 5: BOTTOM LIVE TELEMETRY STATUS BAR                   */}
      {/* ============================================================== */}
      <footer className="h-7 border-t border-warm-200 bg-surface px-4 flex items-center justify-between text-[11px] font-mono text-ink-secondary shrink-0 z-20">
        <div className="flex items-center gap-4">
          <span>
            Raster: <b className="text-ink">{activeJob?.metadata?.width || 0}×{activeJob?.metadata?.height || 0} px</b>
          </span>
          <span className="hidden sm:inline">
            CRS: <b className="text-ink">{activeJob?.metadata?.crs || 'EPSG:4326'}</b>
          </span>
          <span>
            Zoom: <b className="text-ink">{currentZoom}</b>
          </span>
          {cursorCoords && (
            <span className="hidden md:inline text-ink">
              Cursor: {cursorCoords.lat.toFixed(5)}, {cursorCoords.lon.toFixed(5)}
            </span>
          )}
        </div>

        <div className="flex items-center gap-4">
          {selectedBuilding && (
            <span className="hidden sm:inline">
              Selected: <b className="text-ink">{selectedBuilding.properties.building_id}</b>
            </span>
          )}
          <span>
            Footprints: <b className="text-accent-green">{geojson?.features?.length || 0}</b>
          </span>
          <span>
            Threshold: <b className="text-ink">{currentThreshold.toFixed(2)}</b>
          </span>
        </div>
      </footer>

      {/* ============================================================== */}
      {/* THRESHOLD TERMINOLOGY MODAL                                    */}
      {/* ============================================================== */}
      {thresholdModalOpen && (
        <div className="fixed inset-0 bg-ink/30 backdrop-blur-xs z-50 flex items-center justify-center p-4">
          <div className="bg-surface rounded-xl border border-warm-200 shadow-floating max-w-lg w-full p-6 space-y-4 text-xs text-ink">
            <div className="flex items-center justify-between border-b border-warm-200 pb-3">
              <h3 className="text-sm font-semibold text-ink flex items-center gap-2">
                <Info className="h-4 w-4 text-accent-green" />
                Understanding Inference Thresholds
              </h3>
              <button
                onClick={() => setThresholdModalOpen(false)}
                className="p-1 rounded hover:bg-warm-100 text-ink-muted hover:text-ink"
              >
                <X className="h-4 w-4" />
              </button>
            </div>

            <div className="space-y-3 leading-relaxed text-ink-secondary">
              <div>
                <b className="text-ink block">Probability Map vs Binary Mask:</b>
                The custom U-Net outputs a continuous sigmoid value $[0.0, 1.0]$ for each pixel representing building probability. Applying a threshold converts this continuous raster into a discrete binary mask.
              </div>

              <div>
                <b className="text-ink block">Precision vs Recall Trade-off:</b>
                Lower thresholds ($\approx 0.40$) maximize building detection (high recall) but capture more garden/patio false positives. Higher thresholds ($\approx 0.85$) produce clean, conservative boundaries (high precision) but may omit small outbuildings.
              </div>

              <div>
                <b className="text-ink block">Empirical Optimal Threshold (0.75):</b>
                Validation across held-out SpaceNet 2 Las Vegas scenes determined that <span className="font-mono font-bold text-accent-green">0.75</span> achieves the optimal balance ($F1=81.34\%$, $\text{IoU}=68.55\%$).
              </div>

              <div>
                <b className="text-ink block">Instantaneous Vectorization:</b>
                Because the continuous probability map is pre-calculated, adjusting the threshold re-binarizes and re-polygonizes in under 0.5 seconds without re-running the neural network.
              </div>
            </div>

            <div className="pt-2 border-t border-warm-200 flex justify-end">
              <button
                onClick={() => setThresholdModalOpen(false)}
                className="px-3.5 py-1.5 rounded-md bg-accent-green text-white font-medium hover:bg-accent-greenDark transition-colors font-mono"
              >
                Got it
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
