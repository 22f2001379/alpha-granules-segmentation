/**
 * PredictionViewer.jsx
 *
 * React component for visualizing instance segmentation predictions from the platelet-seg API.
 *
 * Features:
 * - File upload and local image path input (dev mode)
 * - Real-time API prediction via POST /api/predict
 * - Display base64 overlay image or draw contours via canvas
 * - Interactive instance table with sorting by area
 * - Zoom and pan controls
 * - Toggle contour visibility
 *
 * Usage:
 *   <PredictionViewer API_BASE="http://localhost:5000" />
 *
 * Props:
 *   - API_BASE (string): Base URL for API endpoints (default: '')
 */

import React, { useState, useRef, useEffect } from 'react';

export default function PredictionViewer({ API_BASE = '' }) {
  // State: input
  const [selectedFile, setSelectedFile] = useState(null);
  const [imagePath, setImagePath] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  // State: prediction results
  const [prediction, setPrediction] = useState(null);
  const [overlayImageUrl, setOverlayImageUrl] = useState(null);
  const [instances, setInstances] = useState([]);

  // State: display options
  const [zoom, setZoom] = useState(1);
  const [showContours, setShowContours] = useState(true);
  const [sortBy, setSortBy] = useState('id'); // 'id' or 'area'
  const [sortOrder, setSortOrder] = useState('asc');

  // Refs
  const fileInputRef = useRef(null);
  const canvasRef = useRef(null);
  const imageRef = useRef(null);
  const containerRef = useRef(null);

  /**
   * Handle file selection from input
   */
  const handleFileChange = (e) => {
    const file = e.target.files[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
    }
  };

  /**
   * Submit prediction request
   */
  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setLoading(true);

    try {
      const formData = new FormData();

      // Use uploaded file or image path
      if (selectedFile) {
        formData.append('image', selectedFile);
      } else if (imagePath) {
        formData.append('image_path', imagePath);
      } else {
        setError('Please select a file or provide an image path.');
        setLoading(false);
        return;
      }

      // Add optional parameters
      formData.append('tile_size', 1024);
      formData.append('overlap', 128);
      formData.append('min_area_px', 50);
      formData.append('overlay_base64', 'true');

      const response = await fetch(`${API_BASE}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (response.ok && data.status === 'success') {
        setPrediction(data);
        setInstances(data.instances || []);

        // Set overlay image if available
        if (data.overlay_base64) {
          setOverlayImageUrl(data.overlay_base64);
        } else {
          setOverlayImageUrl(null);
        }
      } else {
        setError(data.message || 'Prediction failed.');
      }
    } catch (err) {
      setError(`Error: ${err.message}`);
    } finally {
      setLoading(false);
    }
  };

  /**
   * Sort instances by selected column
   */
  const handleSort = (column) => {
    if (sortBy === column) {
      setSortOrder(sortOrder === 'asc' ? 'desc' : 'asc');
    } else {
      setSortBy(column);
      setSortOrder('asc');
    }
  };

  /**
   * Get sorted instances
   */
  const getSortedInstances = () => {
    const sorted = [...instances];
    sorted.sort((a, b) => {
      let aVal = a[sortBy];
      let bVal = b[sortBy];

      if (sortBy === 'area_px') {
        aVal = a.area_px;
        bVal = b.area_px;
      } else if (sortBy === 'centroid') {
        aVal = Math.sqrt(a.centroid[0] ** 2 + a.centroid[1] ** 2);
        bVal = Math.sqrt(b.centroid[0] ** 2 + b.centroid[1] ** 2);
      }

      return sortOrder === 'asc' ? aVal - bVal : bVal - aVal;
    });
    return sorted;
  };

  /**
   * Draw contours on canvas if overlay not available
   */
  useEffect(() => {
    if (!canvasRef.current || !imageRef.current || !showContours) {
      return;
    }

    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');

    // Draw base image
    ctx.drawImage(imageRef.current, 0, 0);

    // Draw contours for each instance
    if (instances.length > 0) {
      instances.forEach((inst, idx) => {
        const [x1, y1, x2, y2] = inst.bbox;
        const color = `hsl(${(idx * 360) / instances.length}, 100%, 50%)`;

        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);

        // Draw centroid
        ctx.fillStyle = color;
        ctx.beginPath();
        ctx.arc(inst.centroid[0], inst.centroid[1], 5, 0, 2 * Math.PI);
        ctx.fill();
      });
    }
  }, [showContours, instances, imageRef]);

  const sortedInstances = getSortedInstances();

  return (
    <div className="min-h-screen bg-gray-100 p-6">
      <div className="max-w-7xl mx-auto">
        {/* Header */}
        <div className="mb-8">
          <h1 className="text-4xl font-bold text-gray-800 mb-2">
            Prediction Viewer
          </h1>
          <p className="text-gray-600">
            Upload an image or provide a local path to run instance segmentation inference.
          </p>
        </div>

        {/* Input Section */}
        <div className="bg-white rounded-lg shadow-md p-6 mb-8">
          <form onSubmit={handleSubmit} className="space-y-4">
            {/* File Upload */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Upload Image
              </label>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileChange}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
              />
              {selectedFile && (
                <p className="text-sm text-gray-500 mt-1">
                  Selected: {selectedFile.name}
                </p>
              )}
            </div>

            {/* Image Path Input (Dev Mode) */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Or Provide Local Image Path (Dev)
              </label>
              <input
                type="text"
                placeholder="/path/to/image.png"
                value={imagePath}
                onChange={(e) => setImagePath(e.target.value)}
                className="w-full px-3 py-2 border border-gray-300 rounded-md shadow-sm focus:ring-blue-500 focus:border-blue-500"
              />
            </div>

            {/* Error Message */}
            {error && (
              <div className="bg-red-50 border border-red-200 rounded-md p-3">
                <p className="text-red-700 text-sm">{error}</p>
              </div>
            )}

            {/* Submit Button */}
            <button
              type="submit"
              disabled={loading || (!selectedFile && !imagePath)}
              className="w-full bg-blue-600 hover:bg-blue-700 disabled:bg-gray-400 text-white font-medium py-2 px-4 rounded-md transition"
            >
              {loading ? 'Running Inference...' : 'Predict'}
            </button>
          </form>
        </div>

        {/* Results Section */}
        {prediction && (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
            {/* Image Viewer */}
            <div className="lg:col-span-2">
              <div className="bg-white rounded-lg shadow-md p-6">
                <div className="flex justify-between items-center mb-4">
                  <h2 className="text-xl font-bold text-gray-800">
                    Prediction Overlay
                  </h2>
                  <div className="flex gap-2">
                    {/* Zoom Controls */}
                    <button
                      onClick={() => setZoom(Math.max(0.5, zoom - 0.1))}
                      className="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded text-sm"
                    >
                      −
                    </button>
                    <span className="px-3 py-1 text-sm font-medium">
                      {(zoom * 100).toFixed(0)}%
                    </span>
                    <button
                      onClick={() => setZoom(Math.min(3, zoom + 0.1))}
                      className="bg-gray-200 hover:bg-gray-300 px-3 py-1 rounded text-sm"
                    >
                      +
                    </button>
                  </div>
                </div>

                {/* Contours Toggle */}
                <div className="flex items-center gap-2 mb-4">
                  <input
                    type="checkbox"
                    id="showContours"
                    checked={showContours}
                    onChange={(e) => setShowContours(e.target.checked)}
                    className="w-4 h-4 text-blue-600 rounded"
                  />
                  <label
                    htmlFor="showContours"
                    className="text-sm font-medium text-gray-700"
                  >
                    Show Contours
                  </label>
                </div>

                {/* Image Container */}
                <div
                  ref={containerRef}
                  className="bg-gray-50 rounded-md overflow-auto max-h-96"
                  style={{ maxHeight: '500px' }}
                >
                  {overlayImageUrl ? (
                    <img
                      src={overlayImageUrl}
                      alt="Prediction Overlay"
                      style={{
                        transform: `scale(${zoom})`,
                        transformOrigin: 'top left',
                        maxWidth: '100%',
                        height: 'auto',
                      }}
                      className="block"
                    />
                  ) : (
                    <canvas
                      ref={canvasRef}
                      style={{
                        transform: `scale(${zoom})`,
                        transformOrigin: 'top left',
                        maxWidth: '100%',
                        height: 'auto',
                      }}
                      className="block"
                    />
                  )}
                </div>
              </div>
            </div>

            {/* Summary & Table */}
            <div className="space-y-6">
              {/* Summary Stats */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">
                  Summary
                </h3>
                <div className="space-y-2">
                  <div>
                    <p className="text-sm text-gray-600">Object Count</p>
                    <p className="text-2xl font-bold text-blue-600">
                      {prediction.summary.count}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Mean Area (px)</p>
                    <p className="text-xl font-semibold text-gray-800">
                      {prediction.summary.mean_area.toFixed(1)}
                    </p>
                  </div>
                  <div>
                    <p className="text-sm text-gray-600">Std Area (px)</p>
                    <p className="text-xl font-semibold text-gray-800">
                      {prediction.summary.std_area.toFixed(1)}
                    </p>
                  </div>
                </div>
              </div>

              {/* Instances Table */}
              <div className="bg-white rounded-lg shadow-md p-6">
                <h3 className="text-lg font-bold text-gray-800 mb-4">
                  Instances
                </h3>
                <div className="overflow-x-auto">
                  <table className="w-full text-sm">
                    <thead>
                      <tr className="border-b">
                        <th
                          onClick={() => handleSort('id')}
                          className="px-2 py-2 text-left font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                        >
                          ID {sortBy === 'id' && (sortOrder === 'asc' ? '↑' : '↓')}
                        </th>
                        <th
                          onClick={() => handleSort('area_px')}
                          className="px-2 py-2 text-left font-semibold text-gray-700 cursor-pointer hover:bg-gray-100"
                        >
                          Area {sortBy === 'area_px' && (sortOrder === 'asc' ? '↑' : '↓')}
                        </th>
                        <th className="px-2 py-2 text-left font-semibold text-gray-700">
                          Centroid
                        </th>
                      </tr>
                    </thead>
                    <tbody>
                      {sortedInstances.map((inst, idx) => (
                        <tr key={idx} className="border-b hover:bg-gray-50">
                          <td className="px-2 py-2 text-gray-800 font-medium">
                            {inst.id}
                          </td>
                          <td className="px-2 py-2 text-gray-800">
                            {inst.area_px}
                          </td>
                          <td className="px-2 py-2 text-gray-600 text-xs">
                            ({inst.centroid[0].toFixed(0)}, {inst.centroid[1].toFixed(0)})
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Empty State */}
        {!prediction && (
          <div className="bg-white rounded-lg shadow-md p-12 text-center">
            <p className="text-gray-500 text-lg">
              No predictions yet. Upload an image and click Predict to get started.
            </p>
          </div>
        )}
      </div>
    </div>
  );
}
