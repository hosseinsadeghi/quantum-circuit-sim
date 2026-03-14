import { useDeferredValue } from 'react';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer, ReferenceLine } from 'recharts';
import { useSimulationContext } from '../context/SimulationContext.jsx';

const THRESHOLD = 0.001;

export default function AmplitudeChart() {
  const { trace, currentStep } = useSimulationContext();
  const deferredStep = useDeferredValue(currentStep);

  if (!trace) return null;

  const step = trace.steps[deferredStep];
  if (!step) return null;

  let chartData = step.probabilities.map((prob, i) => ({
    label: step.basis_labels[i],
    probability: prob,
    realAmp: step.state_vector.real[i],
  }));

  // Collapse negligible states
  const significant = chartData.filter((d) => d.probability >= THRESHOLD);
  const otherProb = chartData.filter((d) => d.probability < THRESHOLD).reduce((s, d) => s + d.probability, 0);

  const displayData = significant.length < chartData.length
    ? [...significant, { label: 'other', probability: otherProb, realAmp: 0 }]
    : chartData;

  return (
    <div className="card">
      <div className="card-title">State Probabilities</div>
      <ResponsiveContainer width="100%" height={220}>
        <BarChart data={displayData} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <XAxis dataKey="label" tick={{ fontSize: 11, fill: '#718096' }} interval={displayData.length > 16 ? 'preserveStartEnd' : 0} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: '#718096' }} />
          <Tooltip
            contentStyle={{ background: '#1e2334', border: '1px solid #2d3748', borderRadius: '6px' }}
            labelStyle={{ color: '#e2e8f0' }}
            formatter={(v) => [v.toFixed(4), 'Probability']}
          />
          <ReferenceLine y={0} stroke="#2d3748" />
          <Bar dataKey="probability" radius={[3, 3, 0, 0]}>
            {displayData.map((entry, idx) => (
              <Cell
                key={idx}
                fill={entry.label === trace.measurement.most_likely_outcome
                  ? '#10b981'
                  : entry.realAmp >= 0 ? '#7c3aed' : '#ef4444'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div style={{ fontSize: '0.75rem', color: '#718096', marginTop: '4px' }}>
        Green = most likely outcome · Purple = positive amplitude · Red = negative amplitude
      </div>
    </div>
  );
}
