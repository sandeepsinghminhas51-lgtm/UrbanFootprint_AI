import React, { useState, useEffect, useRef } from 'react';
import {
  Upload,
  FileCheck,
  AlertTriangle,
  Play,
  Layers,
  Sliders,
  CheckCircle2,
  Clock,
  Loader2,
  Info,
  Maximize2,
  ArrowRight,
  FileUp,
  Image as ImageIcon
} from 'lucide-react';
import { api } from '../api/client';

export default function UploadView({ activeJob, onSelectJob, setCurrentView }) {
  const [dragActive, setDragActive] = useState(false);
  const [uploading, setUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  // Inference tuning options
  const [threshold, setThreshold] = useState(0.75);
  const [overlap, setOverlap] = useState(64);
  const [minPixels, setMinPixels] = useState(3);

  // Execution state
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [analysisError, setAnalysisError] = useState(null);

  const fileInputRef = useRef(null);

  // Poll active job status if it's currently processing
  useEffect(() => {
    let intervalId = null;
    if (activeJob && (activeJob.status === 'processing' || isAnalyzing)) {
      intervalId = setInterval(async () => {
        try {
          const updated = await api.getJobStatus(activeJob.job_id);
          onSelectJob(updated);
          if (updated.status === 'completed') {
            setIsAnalyzing(false);
            clearInterval(intervalId);
          } else if (updated.status === 'failed') {
            setIsAnalyzing(false);
            setAnalysisError(updated.error || 'Inference pipeline failed.');
            clearInterval(intervalId);
          }
        } catch (err) {
          console.error('Polling error:', err);
        }
      }, 1000);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [activeJob, isAnalyzing]);

  const handleFile = async (file) => {
    setUploadError(null);
    setAnalysisError(null);

    const ext = file.name.substring(file.name.lastIndexOf('.')).toLowerCase();
    if (ext !== '.tif' && ext !== '.tiff') {
      setUploadError(
        `Selected file "${file.name}" is not a GeoTIFF. Standard PNG/JPG formats lack spatial Coordinate Reference Systems (CRS) and affine geotransforms. GeoTIFF (.tif/.tiff) is required to calculate real-world building coordinates and square-meter areas.`
      );
      return;
    }

    try {
      setUploading(true);
      const newJob = await api.uploadImage(file);
      onSelectJob(newJob);
    } catch (err) {
      setUploadError(err.message);
    } finally {
      setUploading(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  };

  const startExtraction = async () => {
    if (!activeJob) return;
    try {
      setIsAnalyzing(true);
      setAnalysisError(null);
      const res = await api.startAnalysis(activeJob.job_id, {
        threshold: parseFloat(threshold),
        overlap: parseInt(overlap, 10),
        min_building_pixels: parseInt(minPixels, 10),
      });
      onSelectJob(res);
    } catch (err) {
      setIsAnalyzing(false);
      setAnalysisError(err.message);
    }
  };

  const meta = activeJob?.metadata;

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-8">
      {/* Editorial Header */}
      <div className="space-y-1">
        <h1 className="text-2xl sm:text-3xl font-semibold text-ink font-serif tracking-tight">
          Satellite Image Ingestion & Analysis
        </h1>
        <p className="text-xs sm:text-sm text-ink-secondary">
          Upload any 3-band georeferenced satellite GeoTIFF to run sliding-window U-Net segmentation.
        </p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-8 items-start">
        {/* Left Column: Upload Area & Inference Tuning (5 cols) */}
        <div className="lg:col-span-5 space-y-6">
          {/* Refined Minimal Dropzone */}
          <div
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => fileInputRef.current?.click()}
            className={`border border-dashed rounded-xl p-8 text-center cursor-pointer transition-all flex flex-col items-center justify-center min-h-[220px] ${
              dragActive
                ? 'border-accent-green bg-accent-greenLight/40'
                : 'border-warm-300 hover:border-warm-400 bg-surface'
            }`}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".tif,.tiff"
              onChange={(e) => e.target.files?.[0] && handleFile(e.target.files[0])}
              className="hidden"
            />

            <div className="h-10 w-10 rounded-lg bg-warm-100 text-ink-secondary flex items-center justify-center mb-3">
              {uploading ? (
                <Loader2 className="h-5 w-5 animate-spin text-accent-green" />
              ) : (
                <FileUp className="h-5 w-5 text-accent-green" />
              )}
            </div>

            <div className="space-y-1">
              <span className="text-sm font-medium text-ink">
                {uploading ? 'Processing uploaded GeoTIFF...' : 'Choose a file or drag & drop satellite GeoTIFF'}
              </span>
              <p className="text-xs text-ink-muted">
                Accepts 3-band Pan-Sharpened RGB GeoTIFFs (.tif, .tiff)
              </p>
            </div>
          </div>

          {/* Validation Error Alert */}
          {uploadError && (
            <div className="rounded-lg bg-rose-50 border border-rose-200 p-4 text-xs text-status-error space-y-1">
              <div className="flex items-center gap-2 font-semibold">
                <AlertTriangle className="h-4 w-4" />
                Invalid File Format
              </div>
              <p className="leading-relaxed text-ink-secondary">{uploadError}</p>
            </div>
          )}

          {/* Inference Tuning Card */}
          <div className="rounded-xl bg-surface border border-warm-200 p-6 space-y-5 shadow-subtle">
            <div className="flex items-center justify-between border-b border-warm-200 pb-3">
              <div className="flex items-center gap-2">
                <Sliders className="h-4 w-4 text-accent-green" />
                <h3 className="text-xs font-semibold text-ink uppercase tracking-wider">
                  Inference Tuning
                </h3>
              </div>
              <span className="text-[11px] font-mono text-ink-muted">U-Net Controls</span>
            </div>

            {/* Threshold Slider */}
            <div className="space-y-2">
              <div className="flex justify-between items-center text-xs">
                <span className="text-ink font-medium">Detection Confidence Threshold</span>
                <span className="font-mono text-accent-green font-semibold bg-warm-100 px-2 py-0.5 rounded border border-warm-200">
                  {threshold}
                </span>
              </div>
              <input
                type="range"
                min="0.30"
                max="0.90"
                step="0.05"
                value={threshold}
                onChange={(e) => setThreshold(e.target.value)}
                disabled={isAnalyzing}
                className="w-full accent-accent-green bg-warm-200 h-1.5 rounded cursor-pointer"
              />
              <div className="flex justify-between text-[10px] text-ink-muted">
                <span>0.30 (High Recall)</span>
                <span className="text-accent-green font-medium">0.75 (Optimal F1: 81.3%)</span>
                <span>0.90 (Conservative)</span>
              </div>
            </div>

            {/* Overlap & Min Pixels */}
            <div className="grid grid-cols-2 gap-3 pt-1">
              <div className="space-y-1.5">
                <label className="text-xs text-ink font-medium">Patch Overlap</label>
                <select
                  value={overlap}
                  onChange={(e) => setOverlap(Number(e.target.value))}
                  disabled={isAnalyzing}
                  className="w-full rounded-md bg-warm-50 border border-warm-200 px-2.5 py-1.5 text-xs text-ink font-mono focus:border-accent-green outline-none"
                >
                  <option value={32}>32 px (Fast)</option>
                  <option value={64}>64 px (Default)</option>
                  <option value={96}>96 px (High Density)</option>
                </select>
              </div>

              <div className="space-y-1.5">
                <label className="text-xs text-ink font-medium">Min Footprint Area</label>
                <select
                  value={minPixels}
                  onChange={(e) => setMinPixels(Number(e.target.value))}
                  disabled={isAnalyzing}
                  className="w-full rounded-md bg-warm-50 border border-warm-200 px-2.5 py-1.5 text-xs text-ink font-mono focus:border-accent-green outline-none"
                >
                  <option value={1}>1 px (Unfiltered)</option>
                  <option value={3}>3 px (Default)</option>
                  <option value={8}>8 px (Medium/Large)</option>
                </select>
              </div>
            </div>

            {/* Launch Extraction CTA */}
            {activeJob && (
              <button
                disabled={isAnalyzing || activeJob.status === 'processing'}
                onClick={startExtraction}
                className="w-full py-2.5 rounded-lg text-xs font-semibold flex items-center justify-center gap-2 bg-accent-green hover:bg-accent-greenDark text-white disabled:opacity-50 disabled:cursor-not-allowed shadow-subtle transition-all font-mono"
              >
                {isAnalyzing || activeJob.status === 'processing' ? (
                  <>
                    <Loader2 className="h-4 w-4 animate-spin" />
                    Extracting Footprints...
                  </>
                ) : (
                  <>
                    <Play className="h-4 w-4" />
                    {activeJob.status === 'completed' ? 'Re-Run Footprint Extraction' : 'Start Footprint Extraction'}
                  </>
                )}
              </button>
            )}
          </div>
        </div>

        {/* Right Column: Raster Metadata & Progress Tracker (7 cols) */}
        <div className="lg:col-span-7 space-y-6">
          {!activeJob ? (
            <div className="rounded-xl border border-dashed border-warm-300 bg-surface p-12 text-center space-y-3">
              <Layers className="h-8 w-8 text-ink-muted mx-auto" />
              <h3 className="text-sm font-medium text-ink">No Satellite Raster Selected</h3>
              <p className="text-xs text-ink-secondary max-w-sm mx-auto">
                Upload a satellite GeoTIFF or select one from the test catalog on the dashboard to inspect spatial parameters.
              </p>
            </div>
          ) : (
            <div className="space-y-6">
              {/* Raster Metadata Card */}
              <div className="rounded-xl bg-surface border border-warm-200 p-6 space-y-5 shadow-subtle">
                <div className="flex items-start justify-between gap-4 border-b border-warm-200 pb-4">
                  <div>
                    <span className="text-[10px] font-mono font-medium uppercase px-2 py-0.5 rounded bg-warm-100 text-ink border border-warm-200">
                      GeoTIFF Validated
                    </span>
                    <h2 className="text-base font-semibold text-ink mt-1 font-mono break-all">
                      {activeJob.filename}
                    </h2>
                    <p className="text-xs text-ink-secondary font-mono">{activeJob.job_id}</p>
                  </div>

                  {activeJob.status === 'completed' && (
                    <button
                      onClick={() => setCurrentView('gis')}
                      className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-accent-green text-white hover:bg-accent-greenDark shadow-subtle transition-all shrink-0"
                    >
                      <Layers className="h-3.5 w-3.5" />
                      Open GIS Map
                      <ArrowRight className="h-3.5 w-3.5" />
                    </button>
                  )}
                </div>

                {/* Satellite Thumbnail & Parameters Grid */}
                <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 items-center">
                  {meta?.preview_url && (
                    <div className="sm:col-span-1 rounded-lg overflow-hidden border border-warm-200 bg-warm-100 aspect-square relative">
                      <img
                        src={meta.preview_url}
                        alt="Satellite Preview"
                        className="w-full h-full object-cover"
                      />
                    </div>
                  )}

                  <div className={`space-y-3 ${meta?.preview_url ? 'sm:col-span-2' : 'sm:col-span-3'}`}>
                    <div className="grid grid-cols-2 gap-3 text-xs">
                      <div className="bg-warm-50 p-2.5 rounded border border-warm-200">
                        <span className="text-ink-secondary block text-[11px]">Dimensions</span>
                        <span className="font-mono font-semibold text-ink">
                          {meta?.width} × {meta?.height} px
                        </span>
                      </div>

                      <div className="bg-warm-50 p-2.5 rounded border border-warm-200">
                        <span className="text-ink-secondary block text-[11px]">Bands</span>
                        <span className="font-mono font-semibold text-ink">
                          {meta?.count} Spectral (RGB)
                        </span>
                      </div>

                      <div className="bg-warm-50 p-2.5 rounded border border-warm-200">
                        <span className="text-ink-secondary block text-[11px]">Spatial CRS</span>
                        <span className="font-mono font-semibold text-accent-green">
                          {meta?.crs || 'EPSG:4326'}
                        </span>
                      </div>

                      <div className="bg-warm-50 p-2.5 rounded border border-warm-200">
                        <span className="text-ink-secondary block text-[11px]">Pixel Ground Resolution</span>
                        <span className="font-mono font-semibold text-ink">
                          {meta?.resolution ? `${meta.resolution[0].toExponential(2)}` : '0.3m GSD'}
                        </span>
                      </div>
                    </div>

                    {meta?.wgs84_bounds && (
                      <div className="bg-warm-50 p-2 rounded border border-warm-200 text-xs font-mono">
                        <span className="text-ink-secondary block text-[10px] uppercase">WGS84 Extents</span>
                        <span className="text-ink text-[11px]">
                          [{meta.wgs84_bounds[0][0].toFixed(5)}, {meta.wgs84_bounds[0][1].toFixed(5)}] to [{meta.wgs84_bounds[1][0].toFixed(5)}, {meta.wgs84_bounds[1][1].toFixed(5)}]
                        </span>
                      </div>
                    )}
                  </div>
                </div>
              </div>

              {/* Real Multi-Stage Execution Tracker */}
              <div className="rounded-xl bg-surface border border-warm-200 p-6 space-y-4 shadow-subtle">
                <div className="flex items-center justify-between border-b border-warm-200 pb-3">
                  <div>
                    <h3 className="text-xs font-semibold text-ink uppercase tracking-wider">
                      Processing Pipeline Stages
                    </h3>
                    <p className="text-xs text-ink-secondary">
                      Real-time execution monitoring from radiometric stretch to vectorization.
                    </p>
                  </div>
                  <span className="font-mono font-bold text-sm text-accent-green">
                    {activeJob.progress}%
                  </span>
                </div>

                {/* Progress bar */}
                <div className="w-full bg-warm-100 h-1.5 rounded-full overflow-hidden">
                  <div
                    className={`h-full transition-all duration-300 ${
                      activeJob.status === 'failed' ? 'bg-status-error' : 'bg-accent-green'
                    }`}
                    style={{ width: `${activeJob.progress}%` }}
                  />
                </div>

                {/* Stages List */}
                <div className="space-y-2 pt-1">
                  {activeJob.stages?.map((stage, idx) => (
                    <div
                      key={stage.id}
                      className={`flex items-start gap-3 p-2.5 rounded-md text-xs transition-colors ${
                        stage.status === 'in_progress'
                          ? 'bg-accent-sandLight/50 border border-accent-sand/30'
                          : stage.status === 'completed'
                          ? 'bg-warm-50 border border-warm-200'
                          : stage.status === 'failed'
                          ? 'bg-rose-50 border border-rose-200'
                          : 'opacity-50'
                      }`}
                    >
                      <div className="mt-0.5 shrink-0">
                        {stage.status === 'completed' ? (
                          <CheckCircle2 className="h-4 w-4 text-status-success" />
                        ) : stage.status === 'in_progress' ? (
                          <Loader2 className="h-4 w-4 text-accent-green animate-spin" />
                        ) : stage.status === 'failed' ? (
                          <AlertTriangle className="h-4 w-4 text-status-error" />
                        ) : (
                          <Clock className="h-4 w-4 text-ink-muted" />
                        )}
                      </div>

                      <div className="flex-1 space-y-0.5">
                        <div className="flex items-center justify-between">
                          <span className={`font-medium ${stage.status === 'in_progress' ? 'text-ink font-semibold' : 'text-ink'}`}>
                            {idx + 1}. {stage.label}
                          </span>
                          <span className="font-mono text-[10px] text-ink-muted uppercase">
                            {stage.status}
                          </span>
                        </div>
                        {stage.detail && (
                          <p className="text-[11px] text-ink-secondary font-mono">
                            {stage.detail}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}
                </div>

                {/* Failure Alert */}
                {analysisError && (
                  <div className="rounded-lg bg-rose-50 border border-rose-200 p-3 text-xs text-status-error">
                    <div className="font-semibold flex items-center gap-1.5 mb-1">
                      <AlertTriangle className="h-4 w-4" />
                      Inference Pipeline Failed
                    </div>
                    {analysisError}
                  </div>
                )}

                {/* Finished Notification */}
                {activeJob.status === 'completed' && activeJob.results && (
                  <div className="rounded-lg bg-warm-100/70 border border-warm-200 p-4 flex flex-col sm:flex-row items-center justify-between gap-3">
                    <div className="flex items-center gap-3">
                      <CheckCircle2 className="h-5 w-5 text-status-success shrink-0" />
                      <div>
                        <h4 className="text-xs font-semibold text-ink">Inference Complete</h4>
                        <p className="text-[11px] text-ink-secondary font-mono">
                          {activeJob.results.building_count} footprints detected • {activeJob.results.total_built_area_sqm} m² total area • {activeJob.results.inference_time_seconds}s
                        </p>
                      </div>
                    </div>

                    <button
                      onClick={() => setCurrentView('gis')}
                      className="inline-flex items-center gap-1.5 px-4 py-2 rounded-md text-xs font-semibold bg-accent-green hover:bg-accent-greenDark text-white transition-all shadow-subtle shrink-0"
                    >
                      <Layers className="h-3.5 w-3.5" />
                      Open GIS Map
                    </button>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
