import { useSimulationContext } from '../context/SimulationContext.jsx';

const CATEGORY_LABELS = {
  interference: 'Interference',
  communication: 'Communication',
  variational: 'Variational',
  physics: 'Physics Simulation',
  error_correction: 'Error Correction',
  general: 'General',
};

const CATEGORY_ORDER = ['interference', 'communication', 'variational', 'physics', 'error_correction', 'general'];

export default function AlgorithmSelector() {
  const { algorithms, selectedAlgorithm, selectAlgorithm } = useSimulationContext();

  if (!algorithms.length) return <div className="loading">Loading algorithms...</div>;

  // Group by category
  const grouped = {};
  for (const alg of algorithms) {
    const cat = alg.category || 'general';
    if (!grouped[cat]) grouped[cat] = [];
    grouped[cat].push(alg);
  }

  const categories = CATEGORY_ORDER.filter((c) => grouped[c]);

  return (
    <div className="card">
      <div className="card-title">Algorithm</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '16px' }}>
        {categories.map((cat) => (
          <div key={cat}>
            <div style={{
              fontSize: '0.7rem',
              fontWeight: 700,
              letterSpacing: '0.08em',
              textTransform: 'uppercase',
              color: '#4a5568',
              marginBottom: '6px',
              paddingLeft: '2px',
            }}>
              {CATEGORY_LABELS[cat] || cat}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '6px' }}>
              {grouped[cat].map((alg) => (
                <button
                  key={alg.algorithm_id}
                  onClick={() => selectAlgorithm(alg)}
                  style={{
                    padding: '8px 10px',
                    borderRadius: '6px',
                    border: '1px solid',
                    borderColor: selectedAlgorithm?.algorithm_id === alg.algorithm_id ? '#7c3aed' : '#2d3748',
                    background: selectedAlgorithm?.algorithm_id === alg.algorithm_id ? '#3b1d8a22' : 'transparent',
                    color: '#e2e8f0',
                    cursor: 'pointer',
                    textAlign: 'left',
                    transition: 'all 0.15s',
                  }}
                >
                  <div style={{ fontWeight: 600, fontSize: '0.85rem' }}>{alg.name}</div>
                  <div style={{ fontSize: '0.72rem', color: '#718096', marginTop: '2px', lineHeight: 1.3 }}>
                    {alg.description.slice(0, 65)}…
                  </div>
                </button>
              ))}
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
