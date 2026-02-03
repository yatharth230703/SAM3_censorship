/**
 * JavaScript client for SAM2 segmentation endpoint
 * Works in both Node.js and browser environments
 */

/**
 * Convert File or Blob to base64
 */
async function fileToBase64(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      const base64 = reader.result.split(',')[1]; // Remove data:image/...;base64, prefix
      resolve(base64);
    };
    reader.onerror = reject;
    reader.readAsDataURL(file);
  });
}

/**
 * Segment an image using the Modal endpoint
 */
async function segmentImage(imageFile, options = {}) {
  const {
    endpointUrl,
    pointCoords = null,
    pointLabels = null,
    mode = 'point'
  } = options;

  // Convert image to base64
  const imageBase64 = await fileToBase64(imageFile);

  // Prepare payload
  const payload = {
    image: imageBase64,
    mode: mode
  };

  if (pointCoords) {
    payload.point_coords = pointCoords;
  }
  if (pointLabels) {
    payload.point_labels = pointLabels;
  }

  // Send request
  const response = await fetch(endpointUrl, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(payload)
  });

  if (!response.ok) {
    throw new Error(`HTTP error! status: ${response.status}`);
  }

  return await response.json();
}

/**
 * Example usage in a React component
 */
/*
import React, { useState } from 'react';

function ImageSegmentation() {
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const ENDPOINT_URL = 'https://your-username--sam2-segmentation-segment-endpoint.modal.run';

  const handleImageUpload = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    try {
      // Point-based segmentation
      const result = await segmentImage(file, {
        endpointUrl: ENDPOINT_URL,
        pointCoords: [[100, 100]], // Click coordinates
        pointLabels: [1], // Foreground
        mode: 'point'
      });
      
      setResult(result);
      console.log('Segmentation score:', result.score);
    } catch (error) {
      console.error('Segmentation failed:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleAutoSegment = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    setLoading(true);
    try {
      // Automatic segmentation
      const result = await segmentImage(file, {
        endpointUrl: ENDPOINT_URL,
        mode: 'auto'
      });
      
      setResult(result);
      console.log('Found', result.count, 'objects');
    } catch (error) {
      console.error('Segmentation failed:', error);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div>
      <h2>Image Segmentation</h2>
      <input 
        type="file" 
        accept="image/*" 
        onChange={handleImageUpload}
        disabled={loading}
      />
      <button onClick={handleAutoSegment} disabled={loading}>
        Auto Segment
      </button>
      {loading && <p>Processing...</p>}
      {result && (
        <div>
          <p>Segmentation complete!</p>
          <pre>{JSON.stringify(result, null, 2)}</pre>
        </div>
      )}
    </div>
  );
}

export default ImageSegmentation;
*/

// For Node.js usage
if (typeof module !== 'undefined' && module.exports) {
  module.exports = { segmentImage, fileToBase64 };
}
