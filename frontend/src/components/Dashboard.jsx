import React, { useEffect, useState } from 'react';
import {
  Upload,
  Layers,
  ArrowRight,
  Cpu,
  MapPin,
  CheckCircle2,
  FileText,
  Clock,
  Play,
  Database,
  BarChart3,
  ExternalLink,
  ChevronRight,
  Sparkles
} from 'lucide-react';
import { api } from '../api/client';

export default function Dashboard({ setCurrentView, onSelectJob }) {
  const [samples, setSamples] = useState([]);
  const [recentJobs, setRecentJobs] = useState([]);
  const [loadingSample, setLoadingSample] = useState(false);
  const [loadingJobs, setLoadingJobs] = useState(true);

  useEffect(() => {
    // Load pre-loaded SpaceNet samples
    api.getSamples()
      .then(setSamples)
      .catch((e) => console.log('Samples load error:', e));

    // Load recent analysis jobs
    api.listJobs()
      .then((jobs) => {
        setRecentJobs(jobs);
        setLoadingJobs(false);
      })
      .catch((e) => {
        console.log('Jobs load error:', e);
        setLoadingJobs(false);
      });
  }, []);

  const handleQuickLoad = async (sampleId) => {
    try {
      setLoadingSample(true);
      const job = await api.loadSample(sampleId);
      onSelectJob(job);
      setCurrentView('upload');
    } catch (err) {
      alert(`Failed to load sample: ${err.message}`);
    } finally {
      setLoadingSample(false);
    }
  };

  return (
    <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-10 space-y-10">
      {/* Editorial Welcome & Overview Section */}
      <div className="border border-warm-200 bg-surface rounded-xl p-8 sm:p-10 shadow-subtle space-y-6">
        <div className="max-w-3xl space-y-4">
          <div className="inline-flex items-center gap-2 px-2.5 py-1 rounded text-xs font-mono bg-warm-100 text-ink-secondary border border-warm-200">
            <span>Project SpaceNet 2</span>
            <span>•</span>
            <span>Sub-Meter Aerial Extraction</span>
          </div>

          <h1 className="text-3xl sm:text-4xl font-semibold tracking-tight text-ink font-serif leading-tight">
            Automated Building Footprint Segmentation from Satellite Imagery
          </h1>

          <p className="text-base text-ink-secondary leading-relaxed">
            UrbanFootprint AI is an open geospatial analysis platform running a custom U-Net architecture trained from scratch on SpaceNet 2 30cm pan-sharpened RGB satellite tiles. Extract structural boundaries, calculate metric square-meter areas, and refine building geometries interactively in our GIS explorer.
          </p>

          <div className="flex flex-wrap items-center gap-3 pt-2">
            <button
              onClick={() => setCurrentView('upload')}
              className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium bg-accent-green hover:bg-accent-greenDark text-white shadow-subtle transition-all"
            >
              <Upload className="h-4 w-4" />
              Upload Satellite GeoTIFF
              <ArrowRight className="h-4 w-4" />
            </button>

            {samples.length > 0 && (
              <button
                disabled={loadingSample}
                onClick={() => handleQuickLoad(samples[0].id)}
                className="inline-flex items-center gap-2 px-5 py-2.5 rounded-lg text-sm font-medium bg-surface hover:bg-warm-50 text-ink border border-warm-200 shadow-subtle transition-all"
              >
                <Play className="h-4 w-4 text-accent-green" />
                {loadingSample ? 'Loading Demo...' : 'Load SpaceNet 2 Tile (Img 1)'}
              </button>
            )}

            <button
              onClick={() => setCurrentView('model-info')}
              className="inline-flex items-center gap-1.5 px-4 py-2.5 rounded-lg text-sm font-medium text-ink-secondary hover:text-ink transition-colors"
            >
              <BarChart3 className="h-4 w-4 text-accent-sandDark" />
              Validation Metrics Report
            </button>
          </div>
        </div>
      </div>

      {/* Model Performance Summary Table (Refined, Human-Crafted) */}
      <div className="border border-warm-200 bg-surface rounded-xl p-6 sm:p-8 space-y-6 shadow-subtle">
        <div className="flex flex-col sm:flex-row sm:items-baseline justify-between gap-2 border-b border-warm-200 pb-4">
          <div>
            <h2 className="text-lg font-semibold text-ink tracking-tight">
              Verified Model Validation Statistics
            </h2>
            <p className="text-xs sm:text-sm text-ink-secondary mt-0.5">
              Evaluated on complete held-out SpaceNet 2 Las Vegas 30cm RGB tiles at optimal cutoff threshold <span className="font-mono font-semibold text-ink">0.75</span>.
            </p>
          </div>
          <button
            onClick={() => setCurrentView('model-info')}
            className="text-xs font-medium text-accent-green hover:text-accent-greenDark flex items-center gap-1"
          >
            Detailed Sensitivity Table <ExternalLink className="h-3 w-3" />
          </button>
        </div>

        {/* 4 Clean Metric Blocks */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="p-4 rounded-lg bg-warm-50 border border-warm-200 space-y-1">
            <span className="text-xs text-ink-secondary uppercase tracking-wider font-mono">IoU (Jaccard)</span>
            <div className="text-2xl sm:text-3xl font-semibold text-ink font-mono">68.55%</div>
            <p className="text-[11px] text-ink-muted">Intersection over Union</p>
          </div>

          <div className="p-4 rounded-lg bg-warm-50 border border-warm-200 space-y-1">
            <span className="text-xs text-ink-secondary uppercase tracking-wider font-mono">F1-Score (Dice)</span>
            <div className="text-2xl sm:text-3xl font-semibold text-accent-green font-mono">81.34%</div>
            <p className="text-[11px] text-ink-muted">Harmonic mean of P & R</p>
          </div>

          <div className="p-4 rounded-lg bg-warm-50 border border-warm-200 space-y-1">
            <span className="text-xs text-ink-secondary uppercase tracking-wider font-mono">Precision</span>
            <div className="text-2xl sm:text-3xl font-semibold text-ink font-mono">78.43%</div>
            <p className="text-[11px] text-ink-muted">Confidence specificity</p>
          </div>

          <div className="p-4 rounded-lg bg-warm-50 border border-warm-200 space-y-1">
            <span className="text-xs text-ink-secondary uppercase tracking-wider font-mono">Recall</span>
            <div className="text-2xl sm:text-3xl font-semibold text-ink font-mono">84.48%</div>
            <p className="text-[11px] text-ink-muted">Building coverage</p>
          </div>
        </div>

        <div className="rounded-md bg-warm-100/70 border border-warm-200 p-3.5 text-xs text-ink-secondary flex items-start gap-3">
          <span className="font-semibold text-ink shrink-0">Scientific Disclosure:</span>
          <span className="leading-relaxed">
            These metrics reflect combined validation across held-out full-scale satellite scenes from SpaceNet 2 AOI 2 Vegas. Real-world performance on non-desert topographies, off-nadir sensors, or differing optical resolutions will vary.
          </span>
        </div>
      </div>

      {/* Pre-loaded Satellite Tiles Catalog */}
      {samples.length > 0 && (
        <div className="space-y-4">
          <div className="flex items-center justify-between">
            <div>
              <h2 className="text-lg font-semibold text-ink">Included SpaceNet 2 Test Imagery</h2>
              <p className="text-xs sm:text-sm text-ink-secondary">Select an official 30cm Pan-Sharpened RGB GeoTIFF to test the extraction pipeline immediately.</p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
            {samples.slice(0, 4).map((s) => (
              <div
                key={s.id}
                className="rounded-xl border border-warm-200 bg-surface p-4 hover:border-accent-green/60 hover:shadow-subtle transition-all flex flex-col justify-between space-y-3"
              >
                <div className="space-y-1.5">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-mono font-medium px-2 py-0.5 rounded bg-warm-100 text-ink">
                      {s.display_name}
                    </span>
                    <span className="text-[11px] text-ink-muted font-mono">{s.size_mb} MB</span>
                  </div>
                  <h3 className="text-xs font-medium text-ink truncate" title={s.filename}>{s.filename}</h3>
                  <p className="text-[11px] text-ink-secondary">{s.resolution}</p>
                </div>

                <button
                  disabled={loadingSample}
                  onClick={() => handleQuickLoad(s.id)}
                  className="w-full inline-flex items-center justify-center gap-1.5 px-3 py-2 rounded-md text-xs font-medium bg-warm-50 hover:bg-accent-green hover:text-white text-ink border border-warm-200 hover:border-transparent transition-all"
                >
                  <Play className="h-3.5 w-3.5" />
                  Load for Extraction
                </button>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Recent Analysis Workspace Activity */}
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div>
            <h2 className="text-lg font-semibold text-ink">Recent Analysis Jobs</h2>
            <p className="text-xs sm:text-sm text-ink-secondary">Historical extraction jobs recorded in your local workspace.</p>
          </div>
        </div>

        {loadingJobs ? (
          <div className="text-center py-12 text-ink-muted text-xs font-mono">Loading previous jobs...</div>
        ) : recentJobs.length === 0 ? (
          <div className="rounded-xl border border-dashed border-warm-200 p-10 text-center space-y-2 bg-surface">
            <Clock className="h-6 w-6 text-ink-muted mx-auto" />
            <p className="text-xs text-ink-secondary">No previous jobs found in this workspace yet.</p>
            <button
              onClick={() => setCurrentView('upload')}
              className="text-xs font-medium text-accent-green hover:underline"
            >
              Upload your first satellite image &rarr;
            </button>
          </div>
        ) : (
          <div className="rounded-xl border border-warm-200 bg-surface overflow-hidden divide-y divide-warm-200 shadow-subtle">
            {recentJobs.slice(0, 6).map((job) => (
              <div
                key={job.job_id}
                className="p-4 sm:px-6 flex flex-col sm:flex-row sm:items-center justify-between gap-4 hover:bg-warm-50/60 transition-colors"
              >
                <div className="space-y-1">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-xs font-semibold text-ink">{job.filename}</span>
                    <span
                      className={`text-[10px] font-mono uppercase px-2 py-0.5 rounded ${
                        job.status === 'completed'
                          ? 'bg-accent-greenLight text-accent-greenDark border border-accent-green/20'
                          : job.status === 'processing'
                          ? 'bg-accent-sandLight text-accent-sandDark border border-accent-sand/30 animate-pulse'
                          : 'bg-rose-50 text-status-error border border-rose-200'
                      }`}
                    >
                      {job.status}
                    </span>
                  </div>
                  <div className="flex items-center gap-2 text-xs text-ink-secondary font-mono">
                    <span>{job.job_id}</span>
                    <span>•</span>
                    <span>{new Date(job.created_at).toLocaleString()}</span>
                    {job.results?.building_count !== undefined && (
                      <>
                        <span>•</span>
                        <span className="text-ink font-medium">
                          {job.results.building_count} footprints extracted
                        </span>
                      </>
                    )}
                  </div>
                </div>

                <button
                  onClick={() => {
                    onSelectJob(job);
                    if (job.status === 'completed') {
                      setCurrentView('gis');
                    } else {
                      setCurrentView('upload');
                    }
                  }}
                  className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium bg-surface hover:bg-warm-100 text-ink border border-warm-200 self-start sm:self-auto transition-colors"
                >
                  {job.status === 'completed' ? (
                    <>
                      <Layers className="h-3.5 w-3.5 text-accent-green" />
                      Open GIS Map
                    </>
                  ) : (
                    <>
                      <Play className="h-3.5 w-3.5 text-accent-sandDark" />
                      View Processing Status
                    </>
                  )}
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
