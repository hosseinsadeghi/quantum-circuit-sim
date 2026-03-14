import { BarChart, Bar, XAxis, YAxis, Tooltip, Cell, ResponsiveContainer } from 'recharts';
import { useSimulationContext } from '../context/SimulationContext.jsx';

export default function MeasurementHistogram() {
  const { trace } = useSimulationContext();

  if (!trace) return null;

  const { measurement } = trace;
  const data = measurement.basis_labels.map((label, i) => ({
    label,
    probability: measurement.probabilities[i],
  })).filter((d) => d.probability > 0.001);

  return (
    <div className="card">
      <div className="card-title">Final Measurement Distribution</div>
      <div style={{ fontSize: '0.85rem', color: '#a0aec0', marginBottom: '8px' }}>
        Most likely: <span style={{ color: '#10b981', fontWeight: 600 }}>{measurement.most_likely_outcome}</span>
        {' '}({(measurement.probabilities[measurement.basis_labels.indexOf(measurement.most_likely_outcome)] * 100).toFixed(1)}%)
      </div>
      <ResponsiveContainer width="100%" height={180}>
        <BarChart data={data} margin={{ top: 5, right: 10, left: -10, bottom: 5 }}>
          <XAxis dataKey="label" tick={{ fontSize: 11, fill: '#718096' }} />
          <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: '#718096' }} />
          <Tooltip
            contentStyle={{ background: '#1e2334', border: '1px solid #2d3748', borderRadius: '6px' }}
            formatter={(v) => [(v * 100).toFixed(1) + '%', 'Probability']}
          />
          <Bar dataKey="probability" radius={[3, 3, 0, 0]}>
            {data.map((entry, idx) => (
              <Cell
                key={idx}
                fill={entry.label === measurement.most_likely_outcome ? '#10b981' : '#4f46e5'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
    </div>
  );
}
