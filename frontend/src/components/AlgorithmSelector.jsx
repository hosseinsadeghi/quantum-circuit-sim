import { useSimulationContext } from '../context/SimulationContext.jsx';

export default function AlgorithmSelector() {
  const { algorithms, selectedAlgorithm, selectAlgorithm } = useSimulationContext();

  if (!algorithms.length) return <div className="loading">Loading algorithms...</div>;

  return (
    <div className="card">
      <div className="card-title">Algorithm</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
        {algorithms.map((alg) => (
          <button
            key={alg.algorithm_id}
            onClick={() => selectAlgorithm(alg)}
            style={{
              padding: '10px 12px',
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
            <div style={{ fontWeight: 600, fontSize: '0.9rem' }}>{alg.name}</div>
            <div style={{ fontSize: '0.75rem', color: '#718096', marginTop: '2px' }}>
              {alg.description.slice(0, 70)}...
            </div>
          </button>
        ))}
      </div>
    </div>
  );
}
