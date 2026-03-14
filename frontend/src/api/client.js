const BASE = '/api';

export async function fetchAlgorithms() {
  const res = await fetch(`${BASE}/algorithms`);
  if (!res.ok) throw new Error(`Failed to fetch algorithms: ${res.statusText}`);
  return res.json();
}

export async function runSimulation(algorithmId, parameters) {
  const res = await fetch(`${BASE}/simulate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ algorithm_id: algorithmId, parameters }),
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
