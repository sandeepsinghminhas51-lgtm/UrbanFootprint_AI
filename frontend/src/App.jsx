import React, { useState, useEffect } from 'react';
import Header from './components/Header';
import Dashboard from './components/Dashboard';
import UploadView from './components/UploadView';
import GisViewer from './components/GisViewer';
import ModelInfoView from './components/ModelInfoView';
import { api } from './api/client';

export default function App() {
  const [currentView, setCurrentView] = useState('gis');
  const [activeJob, setActiveJob] = useState(null);
  const [recentJobs, setRecentJobs] = useState([]);

  // Fetch jobs on load
  const loadJobs = async () => {
    try {
      const jobs = await api.listJobs();
      if (jobs && jobs.length > 0) {
        setRecentJobs(jobs);
        const completedJob = jobs.find((j) => j.status === 'completed') || jobs[0];
        setActiveJob(completedJob);
      } else {
        setCurrentView('dashboard');
      }
    } catch (e) {
      console.log('Init jobs list error:', e);
      setCurrentView('dashboard');
    }
  };

  useEffect(() => {
    loadJobs();
  }, []);

  const handleSelectJob = (job) => {
    setActiveJob(job);
    loadJobs(); // refresh list
  };

  return (
    <div className="min-h-screen flex flex-col bg-warm-50 text-ink">
      <Header
        currentView={currentView}
        setCurrentView={setCurrentView}
        activeJob={activeJob}
        onSelectJob={handleSelectJob}
        recentJobs={recentJobs}
      />

      <main className="flex-1 flex flex-col">
        {currentView === 'dashboard' && (
          <Dashboard
            setCurrentView={setCurrentView}
            onSelectJob={handleSelectJob}
          />
        )}

        {currentView === 'upload' && (
          <UploadView
            activeJob={activeJob}
            onSelectJob={handleSelectJob}
            setCurrentView={setCurrentView}
          />
        )}

        {currentView === 'gis' && activeJob && (
          <GisViewer
            activeJob={activeJob}
            setCurrentView={setCurrentView}
          />
        )}

        {currentView === 'model-info' && (
          <ModelInfoView />
        )}
      </main>

      {/* Footer (hidden in GIS Workspace to give full height to the 3-panel map interface) */}
      {currentView !== 'gis' && (
        <footer className="border-t border-warm-200 bg-surface py-5 text-center text-xs text-ink-secondary font-mono">
          <div className="max-w-7xl mx-auto px-4 flex flex-col sm:flex-row items-center justify-between gap-2">
            <span>UrbanFootprint AI • Custom SpaceNet 2 U-Net Segmentation Pipeline</span>
            <span>CPU Compatible • WGS84 GeoJSON Export</span>
          </div>
        </footer>
      )}
    </div>
  );
}
