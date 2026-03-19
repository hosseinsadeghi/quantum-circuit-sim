import fallbackData from './algorithms.json';
import demoTraces from './demo-traces.json';

const BASE = '/api';

let backendAvailable = null;

async function isBackendAvailable() {
  if (backendAvailable !== null) return backendAvailable;
  try {
    const res = await fetch(`${BASE}/health`, { signal: AbortSignal.timeout(2000) });
    backendAvailable = res.ok;
  } catch {
    backendAvailable = false;
  }
  return backendAvailable;
}

export function isDemoMode() {
  return backendAvailable === false;
}

export async function fetchAlgorithms() {
  if (await isBackendAvailable()) {
    const res = await fetch(`${BASE}/algorithms`);
    if (!res.ok) throw new Error(`Failed to fetch algorithms: ${res.statusText}`);
    return res.json();
  }
  return fallbackData;
}

export async function runSimulation(algorithmId, parameters, mode = 'statevector', noiseConfig = null) {
  if (await isBackendAvailable()) {
    const res = await fetch(`${BASE}/simulate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        algorithm_id: algorithmId,
        parameters,
        mode,
        noise_config: noiseConfig,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || res.statusText);
    }
    return res.json();
  }

  // Demo mode: return pre-computed trace if available
  const trace = demoTraces[algorithmId];
  if (trace) return trace;
  throw new Error('No demo data available for this algorithm. Run locally with: npm start');
}

export async function runSweep(algorithmId, fixedParameters, sweepParameter, sweepValues, mode = 'statevector', noiseConfig = null) {
  if (!(await isBackendAvailable())) {
    throw new Error('Sweep requires the backend server. Run locally with: npm start');
  }
  const res = await fetch(`${BASE}/sweep`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      algorithm_id: algorithmId,
      fixed_parameters: fixedParameters,
      sweep_parameter: sweepParameter,
      sweep_values: sweepValues,
      mode,
      noise_config: noiseConfig,
    }),
  });
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }));
    throw new Error(err.detail || res.statusText);
  }
  return res.json();
}

export async function checkHealth() {
  return isBackendAvailable();
}
