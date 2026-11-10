import React, { useEffect, useState } from 'react';
import {
  Layers,
  Satellite,
  Plus,
  ChevronDown,
  FolderOpen,
  Map,
  History,
  Info,
  Sliders,
  CheckCircle2,
  AlertCircle
} from 'lucide-react';
import { api } from '../api/client';

export default function Header({
  currentView,
  setCurrentView,
  activeJob,
  onSelectJob,
  recentJobs = []
}) {
  const [backendStatus, setBackendStatus] = useState('checking'); // 'online' | 'offline' | 'checking'
  const [projectDropdownOpen, setProjectDropdownOpen] = useState(false);

  useEffect(() => {
    const checkHealth = async () => {
      try {
        await api.getHealth();
        setBackendStatus('online');
      } catch (err) {
        setBackendStatus('offline');
      }
    };
    checkHealth();
    const interval = setInterval(checkHealth, 20000);
    return () => clearInterval(interval);
  }, []);

  return (
    <header className="border-b border-warm-200 bg-surface sticky top-0 z-50 select-none">
      <div className="max-w-[1920px] mx-auto px-4 sm:px-6">
        <div className="flex items-center justify-between h-13">
          {/* Left: Brand Identity & Active Project Selector */}
          <div className="flex items-center gap-4">
            <div
              onClick={() => setCurrentView('dashboard')}
              className="flex items-center gap-2.5 cursor-pointer group"
            >
              <div className="h-7 w-7 rounded-md bg-accent-green flex items-center justify-center text-white shadow-subtle group-hover:bg-accent-greenDark transition-colors">
                <Satellite className="h-3.5 w-3.5" />
              </div>
              <div className="flex items-center gap-1.5">
                <span className="text-sm font-semibold tracking-tight text-ink">
                  UrbanFootprint
                </span>
                <span className="text-[10px] font-mono uppercase tracking-wider px-1.5 py-0.2 rounded bg-warm-100 text-ink-secondary border border-warm-200">
                  GIS
                </span>
              </div>
            </div>

            <div className="h-4 w-[1px] bg-warm-200 hidden sm:block" />

            {/* Scene / Project Switcher Dropdown */}
            {activeJob && (
              <div className="relative hidden sm:block">
                <button
                  onClick={() => setProjectDropdownOpen(!projectDropdownOpen)}
                  className="flex items-center gap-2 px-2.5 py-1 rounded-md text-xs font-mono bg-warm-50 hover:bg-warm-100 border border-warm-200 text-ink transition-colors max-w-[280px]"
                >
                  <span className="h-1.5 w-1.5 rounded-full bg-accent-green" />
                  <span className="truncate font-medium">{activeJob.filename}</span>
                  <ChevronDown className="h-3 w-3 text-ink-muted ml-1 shrink-0" />
                </button>

                {projectDropdownOpen && (
                  <div className="absolute left-0 mt-1.5 w-80 rounded-lg bg-surface border border-warm-200 shadow-elevated z-50 py-1.5 text-xs font-mono">
                    <div className="px-3 py-1 text-[10px] text-ink-muted uppercase tracking-wider font-sans border-b border-warm-100">
                      Switch Active Scene
                    </div>
                    <div className="max-h-64 overflow-y-auto divide-y divide-warm-100">
                      {recentJobs.length === 0 ? (
                        <div className="p-3 text-ink-secondary">No other scenes loaded</div>
                      ) : (
                        recentJobs.map((j) => (
                          <div
                            key={j.job_id}
                            onClick={() => {
                              onSelectJob(j);
                              setProjectDropdownOpen(false);
                            }}
                            className={`p-2.5 px-3 hover:bg-warm-50 cursor-pointer flex items-center justify-between ${
                              j.job_id === activeJob.job_id ? 'bg-accent-greenLight/50 text-accent-greenDark font-bold' : 'text-ink'
                            }`}
                          >
                            <span className="truncate">{j.filename}</span>
                            <span className="text-[10px] text-ink-muted shrink-0 ml-2">
                              {j.results?.building_count ? `${j.results.building_count} bldgs` : j.status}
                            </span>
                          </div>
                        ))
                      )}
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>

          {/* Center: Navigation Tabs (Rescope-style) */}
          <nav className="flex items-center gap-1 sm:gap-2">
            <button
              onClick={() => setCurrentView('gis')}
              disabled={!activeJob}
              className={`px-3 py-1.5 rounded-md text-xs font-medium flex items-center gap-1.5 transition-colors ${
                currentView === 'gis'
                  ? 'bg-warm-100 text-ink font-semibold border border-warm-200'
                  : 'text-ink-secondary hover:text-ink hover:bg-warm-50 disabled:opacity-40 disabled:hover:bg-transparent'
              }`}
            >
              <Map className="h-3.5 w-3.5 text-accent-green" />
              <span>Map Workspace</span>
            </button>

            <button
              onClick={() => setCurrentView('upload')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium flex items-center gap-1.5 transition-colors ${
                currentView === 'upload'
                  ? 'bg-warm-100 text-ink font-semibold border border-warm-200'
                  : 'text-ink-secondary hover:text-ink hover:bg-warm-50'
              }`}
            >
              <FolderOpen className="h-3.5 w-3.5 text-accent-sandDark" />
              <span>Upload & Ingest</span>
            </button>

            <button
              onClick={() => setCurrentView('dashboard')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium flex items-center gap-1.5 transition-colors ${
                currentView === 'dashboard'
                  ? 'bg-warm-100 text-ink font-semibold border border-warm-200'
                  : 'text-ink-secondary hover:text-ink hover:bg-warm-50'
              }`}
            >
              <History className="h-3.5 w-3.5 text-ink-muted" />
              <span className="hidden sm:inline">Analysis History</span>
            </button>

            <button
              onClick={() => setCurrentView('model-info')}
              className={`px-3 py-1.5 rounded-md text-xs font-medium flex items-center gap-1.5 transition-colors ${
                currentView === 'model-info'
                  ? 'bg-warm-100 text-ink font-semibold border border-warm-200'
                  : 'text-ink-secondary hover:text-ink hover:bg-warm-50'
              }`}
            >
              <Info className="h-3.5 w-3.5 text-ink-muted" />
              <span className="hidden sm:inline">Model Info</span>
            </button>
          </nav>

          {/* Right: Backend Connection Status & New Ingestion */}
          <div className="flex items-center gap-3">
            <div className="flex items-center gap-2 px-2 py-1 rounded text-xs font-mono bg-warm-50 border border-warm-200 text-ink-secondary">
              <span
                className={`h-2 w-2 rounded-full ${
                  backendStatus === 'online'
                    ? 'bg-status-success'
                    : backendStatus === 'offline'
                    ? 'bg-status-error'
                    : 'bg-status-warning'
                }`}
              />
              <span className="text-[11px] hidden md:inline">
                {backendStatus === 'online' ? 'Backend Connected' : 'Disconnected'}
              </span>
            </div>

            <button
              onClick={() => setCurrentView('upload')}
              className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-md text-xs font-medium bg-accent-green hover:bg-accent-greenDark text-white shadow-subtle transition-colors"
            >
              <Plus className="h-3.5 w-3.5" />
              <span className="hidden sm:inline">Ingest Image</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
}
