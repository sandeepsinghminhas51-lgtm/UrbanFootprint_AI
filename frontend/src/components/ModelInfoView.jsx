import React, { useEffect, useState } from 'react';
import {
  Cpu,
  Layers,
  Database,
  ShieldAlert,
  BarChart2,
  CheckCircle2,
  GitBranch,
  Sliders,
  ExternalLink,
  BookOpen
} from 'lucide-react';
import { api } from '../api/client';

export default function ModelInfoView() {
  const [modelInfo, setModelInfo] = useState(null);

  useEffect(() => {
    api.getModelInfo().then(setModelInfo).catch(console.error);
  }, []);

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      {/* Editorial Header */}
      <div className="space-y-2 border-b border-warm-200 pb-6">
        <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono bg-warm-100 text-ink-secondary border border-warm-200">
          <BookOpen className="h-3.5 w-3.5 text-accent-green" />
          Technical Documentation & Empirical Validation
        </div>
        <h1 className="text-3xl sm:text-4xl font-semibold text-ink font-serif tracking-tight">
          Model Architecture, Training & Validation Transparency
        </h1>
        <p className="text-sm sm:text-base text-ink-secondary max-w-3xl leading-relaxed">
          UrbanFootprint AI implements a custom 4-level U-Net model engineered specifically for sub-meter building footprint segmentation on CPU hardware, avoiding proprietary black-box APIs.
        </p>
      </div>

      {/* 3 Pillars Overview */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="rounded-xl bg-surface border border-warm-200 p-6 space-y-3 shadow-subtle">
          <div className="h-9 w-9 rounded-md bg-warm-100 text-accent-green flex items-center justify-center">
            <Database className="h-4.5 w-4.5" />
          </div>
          <h3 className="text-base font-semibold text-ink">SpaceNet 2 Dataset</h3>
          <p className="text-xs text-ink-secondary leading-relaxed">
            Trained using DigitalGlobe WorldView-3 high-resolution satellite imagery over Las Vegas (AOI 2). Each scene contains 30 cm ground sample distance (GSD) pan-sharpened 3-band RGB with matched vector building outlines.
          </p>
        </div>

        <div className="rounded-xl bg-surface border border-warm-200 p-6 space-y-3 shadow-subtle">
          <div className="h-9 w-9 rounded-md bg-warm-100 text-accent-green flex items-center justify-center">
            <Cpu className="h-4.5 w-4.5" />
          </div>
          <h3 className="text-base font-semibold text-ink">Custom U-Net Architecture</h3>
          <p className="text-xs text-ink-secondary leading-relaxed">
            Trained from scratch without external pre-trained weights. Implemented with DoubleConv residual layers, skip connections, and a 4-level encoder-decoder hierarchy with base channels=16, ensuring fast CPU inference (~3 sec/tile).
          </p>
        </div>

        <div className="rounded-xl bg-surface border border-warm-200 p-6 space-y-3 shadow-subtle">
          <div className="h-9 w-9 rounded-md bg-warm-100 text-accent-green flex items-center justify-center">
            <Layers className="h-4.5 w-4.5" />
          </div>
          <h3 className="text-base font-semibold text-ink">Geospatial Vectorization</h3>
          <p className="text-xs text-ink-secondary leading-relaxed">
            Binary prediction masks are converted to planar polygon features using 4-connectivity shapes extraction. Coordinates are transformed to WGS84 (EPSG:4326), with latitude-scaled geodesic area calculated in square meters.
          </p>
        </div>
      </div>

      {/* Threshold Comparison Table */}
      <div className="rounded-xl bg-surface border border-warm-200 p-6 sm:p-8 space-y-6 shadow-subtle">
        <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 border-b border-warm-200 pb-4">
          <div>
            <h2 className="text-lg font-semibold text-ink">Threshold Sensitivity Analysis</h2>
            <p className="text-xs text-ink-secondary">
              Confusion counts across held-out SpaceNet 2 validation images at varying probability cutoffs.
            </p>
          </div>
          <span className="text-xs font-mono text-ink bg-warm-100 px-2.5 py-1 rounded border border-warm-200 self-start sm:self-auto">
            Selected Optimal Cutoff: 0.75
          </span>
        </div>

        <div className="overflow-x-auto">
          <table className="w-full text-left text-xs font-mono">
            <thead>
              <tr className="border-b border-warm-200 text-ink-secondary">
                <th className="py-2.5 px-3">Threshold</th>
                <th className="py-2.5 px-3">Precision</th>
                <th className="py-2.5 px-3">Recall</th>
                <th className="py-2.5 px-3">F1-Score</th>
                <th className="py-2.5 px-3">IoU (Jaccard)</th>
                <th className="py-2.5 px-3">Trade-off Profile</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-warm-100">
              {modelInfo?.comparison_samples?.map((row) => {
                const isOptimal = row.threshold === 0.75;
                return (
                  <tr
                    key={row.threshold}
                    className={isOptimal ? 'bg-accent-greenLight/60 font-semibold text-ink' : 'text-ink-secondary hover:bg-warm-50'}
                  >
                    <td className="py-2.5 px-3 flex items-center gap-1.5">
                      {row.threshold.toFixed(2)}
                      {isOptimal && (
                        <span className="text-[10px] bg-accent-green text-white px-1.5 py-0.2 rounded font-sans font-bold">
                          Optimal
                        </span>
                      )}
                    </td>
                    <td className="py-2.5 px-3">{(row.precision * 100).toFixed(2)}%</td>
                    <td className="py-2.5 px-3">{(row.recall * 100).toFixed(2)}%</td>
                    <td className="py-2.5 px-3">{(row.f1 * 100).toFixed(2)}%</td>
                    <td className="py-2.5 px-3">{(row.iou * 100).toFixed(2)}%</td>
                    <td className="py-2.5 px-3 text-ink-secondary font-sans">
                      {row.threshold < 0.6
                        ? 'High recall, higher false alarms'
                        : isOptimal
                        ? 'Optimal balance (81.3% F1, 68.6% IoU)'
                        : 'Conservative boundary delineation'}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Disclaimers & Operational Boundaries */}
      <div className="rounded-xl bg-warm-100/70 border border-warm-200 p-6 sm:p-8 space-y-4">
        <div className="flex items-center gap-2.5 text-ink font-semibold text-sm">
          <ShieldAlert className="h-4.5 w-4.5 text-accent-sandDark" />
          Model Scope, Generalization Limitations & Disclaimers
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6 text-xs text-ink-secondary leading-relaxed">
          <div className="space-y-1.5">
            <h4 className="font-semibold text-ink">Geographic & Sensor Constraints</h4>
            <p>
              The model was trained exclusively on SpaceNet 2 Pan-Sharpened RGB tiles over Las Vegas, NV. Structural geometries in desert suburban developments differ substantially from dense multi-story European or Southeast Asian metropolitan areas.
            </p>
          </div>

          <div className="space-y-1.5">
            <h4 className="font-semibold text-ink">Resolution & Off-Nadir Angles</h4>
            <p>
              Performance relies on ~30cm ground resolution. Low-resolution sensors (e.g. Sentinel-2 10m, Landsat 30m) cannot resolve individual residential roofs. High off-nadir view angles cause building facade lean that may displace rooftop footprints from ground coordinates.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
