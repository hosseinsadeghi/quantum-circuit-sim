import { useSimulationContext } from '../context/SimulationContext.jsx';
import { useStepPlayer } from '../hooks/useStepPlayer.js';

export default function StepScrubber() {
  const { trace, currentStep, setCurrentStep } = useSimulationContext();
  const totalSteps = trace?.steps?.length ?? 0;
  const { playing, play, stop, stepForward, stepBackward, speed, setSpeed } = useStepPlayer(
    totalSteps,
    currentStep,
    setCurrentStep
  );

  if (!trace) return null;

  const step = trace.steps[currentStep];

  return (
    <div className="card" style={{ gridColumn: '1 / -1' }}>
      <div className="card-title">Step Player</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '12px', flexWrap: 'wrap' }}>
        <button className="btn btn-secondary" onClick={stepBackward} title="Previous (←)">◀</button>
        <button className="btn btn-primary" onClick={playing ? stop : play} style={{ minWidth: '80px' }}>
          {playing ? 'Pause' : 'Play'}
        </button>
        <button className="btn btn-secondary" onClick={stepForward} title="Next (→)">▶</button>
        <input
          type="range"
          min={0}
          max={totalSteps - 1}
          value={currentStep}
          onChange={(e) => { stop(); setCurrentStep(Number(e.target.value)); }}
          style={{ flex: 1, minWidth: '120px', accentColor: '#7c3aed' }}
        />
        <span style={{ fontSize: '0.85rem', color: '#a0aec0', whiteSpace: 'nowrap' }}>
          {currentStep + 1} / {totalSteps}
        </span>
        <select
          value={speed}
          onChange={(e) => setSpeed(Number(e.target.value))}
          style={{ background: '#0f1117', border: '1px solid #2d3748', color: '#e2e8f0', borderRadius: '4px', padding: '4px' }}
        >
          <option value={1500}>0.5x</option>
          <option value={800}>1x</option>
          <option value={400}>2x</option>
          <option value={200}>4x</option>
        </select>
      </div>
      {step && (
        <div style={{ fontSize: '0.9rem', color: '#a0aec0' }}>
          <span style={{ color: '#7c3aed', fontWeight: 600 }}>{step.gate ?? 'Init'}</span>
          {' — '}
          {step.label}
        </div>
      )}
    </div>
  );
}
