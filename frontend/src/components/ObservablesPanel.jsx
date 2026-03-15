/**
 * ObservablesPanel — shows per-step observables: purity, entropy, ⟨Z⟩ per qubit.
 */
import { useSimulationContext } from '../context/SimulationContext.jsx';

function Bar({ value, max = 1, color = '#7c3aed' }) {
  const pct = Math.max(0, Math.min(1, value / max)) * 100;
  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: '8px', width: '100%' }}>
      <div style={{
        flex: 1, height: '8px', background: '#1a202c', borderRadius: '4px', overflow: 'hidden',
      }}>
        <div style={{ width: `${pct}%`, height: '100%', background: color, borderRadius: '4px', transition: 'width 0.2s' }} />
      </div>
      <span style={{ fontSize: '0.75rem', color: '#a0aec0', minWidth: '42px', textAlign: 'right' }}>
        {value.toFixed(3)}
      </span>
    </div>
  );
}

export default function ObservablesPanel() {
  const { trace, currentStep } = useSimulationContext();
  if (!trace) return null;

  const step = trace.steps[currentStep];
  if (!step?.observables) return null;

  const { bloch_vectors, z_expectations, entanglement_entropy, purity } = step.observables;

  return (
    <div className="card">
      <div className="card-title">Observables — Step {currentStep}</div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>

        <div>
          <div style={{ fontSize: '0.78rem', color: '#a0aec0', marginBottom: '4px' }}>
            Purity Tr(ρ²)
          </div>
          <Bar value={purity} max={1} color="#10b981" />
        </div>

        {trace.n_qubits >= 2 && (
          <div>
            <div style={{ fontSize: '0.78rem', color: '#a0aec0', marginBottom: '4px' }}>
              Entanglement Entropy (ebits)
            </div>
            <Bar value={entanglement_entropy} max={Math.floor(trace.n_qubits / 2)} color="#f59e0b" />
          </div>
        )}

        <div>
          <div style={{ fontSize: '0.78rem', color: '#a0aec0', marginBottom: '6px' }}>
            ⟨Z⟩ per qubit (−1 = |1⟩, +1 = |0⟩)
          </div>
          {z_expectations.map((z, qi) => (
            <div key={qi} style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '4px' }}>
              <span style={{ fontSize: '0.75rem', color: '#718096', minWidth: '22px' }}>q{qi}</span>
              {/* Map [-1,1] → [0,100%] */}
              <Bar value={(z + 1) / 2} max={1} color="#7c3aed" />
              <span style={{ fontSize: '0.72rem', color: '#718096', minWidth: '40px' }}>
                {z >= 0 ? '+' : ''}{z.toFixed(2)}
              </span>
            </div>
          ))}
        </div>

      </div>
    </div>
  );
}
