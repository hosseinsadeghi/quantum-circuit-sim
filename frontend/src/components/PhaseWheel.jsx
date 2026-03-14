import { useSimulationContext } from '../context/SimulationContext.jsx';

const RADIUS = 24;
const SIZE = RADIUS * 2 + 10;

function SingleWheel({ real, imag, label, isLikely }) {
  const angle = Math.atan2(imag, real);
  const magnitude = Math.sqrt(real * real + imag * imag);
  const ex = RADIUS + RADIUS * magnitude * Math.cos(angle);
  const ey = RADIUS - RADIUS * magnitude * Math.sin(angle);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '2px' }}>
      <svg width={SIZE} height={SIZE}>
        <circle cx={RADIUS} cy={RADIUS} r={RADIUS} fill="none" stroke="#2d3748" strokeWidth={1} />
        <line x1={RADIUS} y1={RADIUS} x2={ex} y2={ey}
          stroke={isLikely ? '#10b981' : '#7c3aed'} strokeWidth={2} strokeLinecap="round" />
        <circle cx={RADIUS} cy={RADIUS} r={2} fill="#a0aec0" />
      </svg>
      <span style={{ fontSize: '9px', color: '#718096' }}>{label}</span>
    </div>
  );
}

export default function PhaseWheel() {
  const { trace, currentStep, showPhaseWheel } = useSimulationContext();

  if (!trace || !showPhaseWheel) return null;

  const step = trace.steps[currentStep];
  if (!step) return null;

  const significantStates = step.probabilities
    .map((p, i) => ({ p, i }))
    .filter(({ p }) => p > 0.001)
    .slice(0, 32); // Limit to 32 for performance

  return (
    <div className="card">
      <div className="card-title">Phase Wheels</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '8px' }}>
        {significantStates.map(({ i }) => (
          <SingleWheel
            key={i}
            real={step.state_vector.real[i]}
            imag={step.state_vector.imag[i]}
            label={step.basis_labels[i]}
            isLikely={step.basis_labels[i] === trace.measurement.most_likely_outcome}
          />
        ))}
      </div>
    </div>
  );
}
