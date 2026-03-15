/**
 * BlochSphere — SVG projection of the Bloch sphere for a single qubit.
 *
 * Uses a simple orthographic projection: draw the equator as an ellipse,
 * then plot the state vector as a line from center to surface.
 */
import { useSimulationContext } from '../context/SimulationContext.jsx';

const R = 55;           // sphere radius in px
const CX = 70;          // center x
const CY = 70;          // center y
const SVG_SIZE = 140;

function project(x, y, z) {
  // Orthographic: x→right, z→up, y→depth (foreshortened by 0.5)
  return {
    px: CX + R * (x - 0.5 * y),
    py: CY - R * (z - 0.0 * y),
  };
}

function BlochSphereQubit({ bloch, label }) {
  const [bx, by, bz] = bloch;
  const tip = project(bx, by, bz);

  // Equator ellipse axes
  const rx = R;
  const ry = R * 0.35;

  // North and South pole
  const north = project(0, 0, 1);
  const south = project(0, 0, -1);

  const xAxisTip = project(1, 0, 0);
  const yAxisTip = project(0, 1, 0);

  return (
    <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '4px' }}>
      <svg width={SVG_SIZE} height={SVG_SIZE + 10} style={{ overflow: 'visible' }}>
        {/* Sphere outline */}
        <circle cx={CX} cy={CY} r={R} fill="none" stroke="#2d3748" strokeWidth={1} />

        {/* Equator */}
        <ellipse cx={CX} cy={CY} rx={rx} ry={ry} fill="none" stroke="#2d3748" strokeWidth={1} strokeDasharray="4 3" />

        {/* Vertical axis */}
        <line x1={CX} y1={CY - R - 8} x2={CX} y2={CY + R + 8} stroke="#2d3748" strokeWidth={1} />

        {/* Pole labels */}
        <text x={CX} y={CY - R - 10} textAnchor="middle" fill="#718096" fontSize={9}>|0⟩</text>
        <text x={CX} y={CY + R + 20} textAnchor="middle" fill="#718096" fontSize={9}>|1⟩</text>

        {/* X axis hint */}
        <line x1={CX} y1={CY} x2={xAxisTip.px} y2={xAxisTip.py} stroke="#2d374855" strokeWidth={1} />

        {/* State vector */}
        <line
          x1={CX} y1={CY}
          x2={tip.px} y2={tip.py}
          stroke="#7c3aed"
          strokeWidth={2.5}
          strokeLinecap="round"
        />

        {/* Tip dot */}
        <circle cx={tip.px} cy={tip.py} r={5} fill="#7c3aed" />

        {/* Coordinates */}
        <text x={CX} y={SVG_SIZE + 8} textAnchor="middle" fill="#a0aec0" fontSize={8}>
          x={bx.toFixed(2)} y={by.toFixed(2)} z={bz.toFixed(2)}
        </text>
      </svg>
      <div style={{ fontSize: '0.75rem', color: '#718096' }}>{label}</div>
    </div>
  );
}

export default function BlochSphere() {
  const { trace, currentStep } = useSimulationContext();
  if (!trace) return null;

  const step = trace.steps[currentStep];
  if (!step?.observables?.bloch_vectors) return null;

  const blochVectors = step.observables.bloch_vectors;

  return (
    <div className="card">
      <div className="card-title">Bloch Spheres</div>
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: '12px', justifyContent: 'center' }}>
        {blochVectors.map((bv, qi) => (
          <BlochSphereQubit key={qi} bloch={bv} label={`q${qi}`} />
        ))}
      </div>
    </div>
  );
}
