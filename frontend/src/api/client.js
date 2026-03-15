const BASE = '/api';

export async function fetchAlgorithms() {
  const res = await fetch(`${BASE}/algorithms`);
  if (!res.ok) throw new Error(`Failed to fetch algorithms: ${res.statusText}`);
  return res.json();
}

export async function runSimulation(algorithmId, parameters, mode = 'statevector', noiseConfig = null) {
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

export async function runSweep(algorithmId, fixedParameters, sweepParameter, sweepValues, mode = 'statevector', noiseConfig = null) {
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
  const res = await fetch(`${BASE}/health`);
  return res.ok;
}
