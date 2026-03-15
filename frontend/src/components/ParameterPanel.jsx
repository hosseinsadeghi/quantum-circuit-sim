import { useSimulationContext } from '../context/SimulationContext.jsx';

export default function ParameterPanel() {
  const { selectedAlgorithm, parameters, setParameters, simulate, loading } = useSimulationContext();

  if (!selectedAlgorithm) return null;

  const schema = selectedAlgorithm.parameter_schema;
  const props = schema?.properties || {};
  const hasParams = Object.keys(props).length > 0;

  const handleChange = (key, value) => {
    setParameters((prev) => ({ ...prev, [key]: value }));
  };

  return (
    <div className="card">
      <div className="card-title">Parameters</div>
      {hasParams && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '12px', marginBottom: '16px' }}>
          {Object.entries(props).map(([key, def]) => (
            <div key={key}>
              <label style={{ display: 'block', fontSize: '0.8rem', color: '#a0aec0', marginBottom: '4px' }}>
                {def.description || key}
              </label>
              {def.enum ? (
                <select
                  value={parameters[key] ?? def.default ?? ''}
                  onChange={(e) => {
                    const raw = e.target.value;
                    handleChange(key, def.type === 'integer' ? parseInt(raw, 10) : raw);
                  }}
                  style={inputStyle}
                >
                  {def.enum.map((opt) => (
                    <option key={opt} value={opt}>{opt}</option>
                  ))}
                </select>
              ) : def.type === 'integer' ? (
                <input
                  type="number"
                  value={parameters[key] ?? def.default ?? ''}
                  min={def.minimum}
                  max={def.maximum}
                  onChange={(e) => handleChange(key, parseInt(e.target.value, 10))}
                  style={inputStyle}
                />
              ) : def.type === 'number' ? (
                <input
                  type="number"
                  value={parameters[key] ?? def.default ?? ''}
                  min={def.minimum}
                  max={def.maximum}
                  step="0.01"
                  onChange={(e) => handleChange(key, parseFloat(e.target.value))}
                  style={inputStyle}
                />
              ) : (
                <input
                  type="text"
                  value={parameters[key] ?? ''}
                  onChange={(e) => handleChange(key, e.target.value)}
                  placeholder={def.pattern ? `Pattern: ${def.pattern}` : ''}
                  style={inputStyle}
                />
              )}
            </div>
          ))}
        </div>
      )}
      {!hasParams && (
        <p style={{ fontSize: '0.85rem', color: '#718096', marginBottom: '16px' }}>
          No parameters required.
        </p>
      )}
      <button
        className="btn btn-primary"
        onClick={simulate}
        disabled={loading}
        style={{ width: '100%' }}
      >
        {loading ? 'Simulating...' : 'Run Simulation'}
      </button>
    </div>
  );
}

const inputStyle = {
  width: '100%',
  padding: '8px 10px',
  background: '#0f1117',
  border: '1px solid #2d3748',
  borderRadius: '5px',
  color: '#e2e8f0',
  fontSize: '0.9rem',
};
