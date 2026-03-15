import { useSimulationContext } from '../context/SimulationContext.jsx';

const CHANNEL_DEFAULTS = {
  depolarizing: { type: 'depolarizing', p: 0.01 },
  amplitude_damping: { type: 'amplitude_damping', gamma: 0.05 },
  phase_damping: { type: 'phase_damping', gamma: 0.05 },
};

const inputStyle = {
  width: '80px',
  padding: '4px 6px',
  background: '#0f1117',
  border: '1px solid #2d3748',
  borderRadius: '4px',
  color: '#e2e8f0',
  fontSize: '0.85rem',
};

const selectStyle = {
  ...inputStyle,
  width: '160px',
};

export default function NoisePanel() {
  const { mode, setMode, noiseConfig, setNoiseConfig } = useSimulationContext();

  const channelType = noiseConfig?.gate_noise?.default?.type ?? 'none';
  const channelParam = noiseConfig?.gate_noise?.default
    ? Object.entries(noiseConfig.gate_noise.default).find(([k]) => k !== 'type')
    : null;

  const handleModeChange = (e) => {
    setMode(e.target.value);
  };

  const handleChannelChange = (e) => {
    const ch = e.target.value;
    if (ch === 'none') {
      setNoiseConfig(null);
      setMode('statevector');
    } else {
      setNoiseConfig({ gate_noise: { default: { ...CHANNEL_DEFAULTS[ch] } } });
      setMode('density_matrix');
    }
  };

  const handleParamChange = (paramKey, value) => {
    setNoiseConfig((prev) => ({
      ...prev,
      gate_noise: {
        ...prev.gate_noise,
        default: { ...prev.gate_noise.default, [paramKey]: parseFloat(value) },
      },
    }));
  };

  const paramEntry = noiseConfig?.gate_noise?.default
    ? Object.entries(noiseConfig.gate_noise.default).find(([k]) => k !== 'type')
    : null;

  return (
    <div className="card">
      <div className="card-title">Noise & Mode</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>

        <div>
          <label style={{ fontSize: '0.78rem', color: '#a0aec0', display: 'block', marginBottom: '4px' }}>
            Simulation Mode
          </label>
          <select value={mode} onChange={handleModeChange} style={selectStyle}>
            <option value="statevector">Statevector (pure)</option>
            <option value="density_matrix">Density Matrix</option>
          </select>
        </div>

        <div>
          <label style={{ fontSize: '0.78rem', color: '#a0aec0', display: 'block', marginBottom: '4px' }}>
            Noise Channel
          </label>
          <select value={channelType} onChange={handleChannelChange} style={selectStyle}>
            <option value="none">No noise</option>
            <option value="depolarizing">Depolarizing</option>
            <option value="amplitude_damping">Amplitude Damping (T1)</option>
            <option value="phase_damping">Phase Damping (T2)</option>
          </select>
        </div>

        {paramEntry && (
          <div>
            <label style={{ fontSize: '0.78rem', color: '#a0aec0', display: 'block', marginBottom: '4px' }}>
              {paramEntry[0] === 'p' ? 'Error probability p' : `${paramEntry[0]} (strength)`}
            </label>
            <input
              type="number"
              min="0"
              max={paramEntry[0] === 'p' ? '0.75' : '1.0'}
              step="0.005"
              value={paramEntry[1]}
              onChange={(e) => handleParamChange(paramEntry[0], e.target.value)}
              style={inputStyle}
            />
          </div>
        )}

        {noiseConfig && (
          <div style={{ fontSize: '0.73rem', color: '#f59e0b', padding: '4px 0' }}>
            ⚠ Noise forces density matrix mode
          </div>
        )}
      </div>
    </div>
  );
}
