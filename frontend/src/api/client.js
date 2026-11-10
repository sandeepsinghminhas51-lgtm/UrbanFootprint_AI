const API_BASE = import.meta.env.VITE_API_BASE_URL || '';

async function request(endpoint, options = {}) {
  const url = `${API_BASE}${endpoint}`;
  try {
    const res = await fetch(url, options);
    if (!res.ok) {
      let errorMessage = `HTTP ${res.status}: ${res.statusText}`;
      try {
        const errJson = await res.json();
        if (errJson.detail) {
          errorMessage = typeof errJson.detail === 'string' ? errJson.detail : JSON.stringify(errJson.detail);
        }
      } catch (_) {}
      throw new Error(errorMessage);
    }
    return await res.json();
  } catch (err) {
    console.error(`API Error on [${endpoint}]:`, err);
    throw err;
  }
}

export const api = {
  // Health & Model
  getHealth: () => request('/api/health'),
  getModelInfo: () => request('/api/model/info'),

  // Samples
  getSamples: () => request('/api/samples'),
  loadSample: (sampleId) =>
    request(`/api/samples/load/${sampleId}`, { method: 'POST' }),

  // Upload
  uploadImage: async (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return request('/api/upload', {
      method: 'POST',
      body: formData,
    });
  },

  // Jobs & Inference
  startAnalysis: (jobId, options = {}) =>
    request(`/api/analyze/${jobId}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        threshold: options.threshold ?? 0.75,
        overlap: options.overlap ?? 64,
        patch_size: options.patch_size ?? 256,
        min_building_pixels: options.min_building_pixels ?? 3,
      }),
    }),

  listJobs: () => request('/api/jobs'),
  getJobStatus: (jobId) => request(`/api/jobs/${jobId}`),
  getJobResults: (jobId) => request(`/api/jobs/${jobId}/results`),

  // Dynamic Threshold Adjustment (No U-Net re-inference)
  applyThreshold: (jobId, threshold, minBuildingPixels = 3) =>
    request(`/api/jobs/${jobId}/threshold`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ threshold, min_building_pixels: minBuildingPixels }),
    }),

  // Ground Truth
  getGroundTruth: (jobId) => request(`/api/jobs/${jobId}/groundtruth`),

  // Buildings & GeoJSON
  getBuildings: (jobId) => request(`/api/jobs/${jobId}/buildings`),

  updateBuilding: (jobId, buildingId, data) =>
    request(`/api/jobs/${jobId}/buildings/${buildingId}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(data),
    }),

  deleteBuilding: (jobId, buildingId) =>
    request(`/api/jobs/${jobId}/buildings/${buildingId}`, {
      method: 'DELETE',
    }),

  getExportGeoJsonUrl: (jobId) => `${API_BASE}/api/jobs/${jobId}/export/geojson`,
  getExportSingleBuildingUrl: (jobId, buildingId) =>
    `${API_BASE}/api/jobs/${jobId}/buildings/${buildingId}/export`,
  getDownloadUrl: (jobId, fileType) => `${API_BASE}/api/jobs/${jobId}/download/${fileType}`,
};
