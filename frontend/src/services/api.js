import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

const api = axios.create({
  baseURL: API_BASE_URL,
  timeout: 120000,
  headers: {
    'Content-Type': 'application/json',
  },
});

// Predict trajectory
export const predictTrajectory = async (sequence) => {
  const response = await api.post('/predict', { sequence });
  return response.data;
};

// Safety audit
export const runSafetyAudit = async (sequence, predictedTrajectory) => {
  const response = await api.post('/agent/safety-audit', {
    sequence,
    predicted_trajectory: predictedTrajectory,
  });
  return response.data;
};

// Driver profile
export const runDriverProfile = async (sequence, predictedTrajectory) => {
  const response = await api.post('/agent/driver-profile', {
    sequence,
    predicted_trajectory: predictedTrajectory,
  });
  return response.data;
};

export default api;
