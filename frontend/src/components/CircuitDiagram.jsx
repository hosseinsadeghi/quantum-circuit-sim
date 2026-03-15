import { useRef, useEffect } from 'react';
import * as d3 from 'd3';
import { useSimulationContext } from '../context/SimulationContext.jsx';

const QUBIT_HEIGHT = 50;
const COL_WIDTH = 80;
const MARGIN = { top: 30, right: 20, bottom: 20, left: 60 };
const GATE_SIZE = 38;

// Gates that need a wider box because their label is long
const WIDE_GATES = new Set(['Oracle', 'Diffusion', 'Measure', 'Reset', 'Barrier']);
const GATE_W = (name) => WIDE_GATES.has(name) ? 58 : GATE_SIZE;
const GATE_LABEL = (name) => name.length > 9 ? name.slice(0, 8) + '…' : name;

const GATE_COLORS = {
  H: '#7c3aed',
  X: '#ef4444',
  Y: '#f59e0b',
  Z: '#3b82f6',
  S: '#06b6d4',
  T: '#0ea5e9',
  CNOT_ctrl: '#10b981',
  CNOT_tgt: '#10b981',
  CZ_ctrl: '#10b981',
  CZ_tgt: '#10b981',
  Oracle: '#f59e0b',
  Diffusion: '#ec4899',
  Rz: '#06b6d4',
  Rx: '#8b5cf6',
  Ry: '#6366f1',
  Measure: '#f97316',
  Reset: '#6b7280',
  Barrier: '#2d3748',
  CP: '#14b8a6',
  'CP†': '#14b8a6',
  ZZ: '#a855f7',
  SWAP_1: '#64748b',
  SWAP_2: '#64748b',
  P: '#0ea5e9',
  default: '#4a5568',
};

export default function CircuitDiagram() {
  const svgRef = useRef(null);
  const containerRef = useRef(null);
  const { trace, currentStep } = useSimulationContext();

  useEffect(() => {
    if (!trace || !svgRef.current) return;

    const { circuit_layout } = trace;
    const n_qubits = circuit_layout.qubit_labels.length;
    const n_cols = circuit_layout.columns.length;

    const width = MARGIN.left + n_cols * COL_WIDTH + MARGIN.right;
    const height = MARGIN.top + n_qubits * QUBIT_HEIGHT + MARGIN.bottom;

    const svg = d3.select(svgRef.current);
    svg.selectAll('*').remove();
    svg.attr('width', width).attr('height', height);

    const g = svg.append('g').attr('transform', `translate(${MARGIN.left},${MARGIN.top})`);

    // Qubit wires
    circuit_layout.qubit_labels.forEach((label, qi) => {
      const y = qi * QUBIT_HEIGHT + QUBIT_HEIGHT / 2;
      g.append('line')
        .attr('x1', -MARGIN.left + 10).attr('x2', n_cols * COL_WIDTH + MARGIN.right)
        .attr('y1', y).attr('y2', y)
        .attr('stroke', '#2d3748').attr('stroke-width', 1.5);

      g.append('text')
        .attr('x', -MARGIN.left + 8).attr('y', y + 4)
        .attr('fill', '#718096').attr('font-size', 12)
        .attr('text-anchor', 'start')
        .text(label);
    });

    // Find which column index corresponds to currentStep
    const activeColIdx = circuit_layout.columns.findIndex(
      (col) => col.gates.some((gate) => gate.step_index === currentStep)
    );

    // Auto-scroll container to keep active column centered
    if (activeColIdx >= 0 && containerRef.current) {
      const columnCenterX = MARGIN.left + activeColIdx * COL_WIDTH + COL_WIDTH / 2;
      const containerWidth = containerRef.current.clientWidth;
      containerRef.current.scrollTo({
        left: columnCenterX - containerWidth / 2,
        behavior: 'smooth',
      });
    }

    // Playhead highlight
    if (activeColIdx >= 0) {
      g.append('rect')
        .attr('x', activeColIdx * COL_WIDTH - COL_WIDTH / 2 + 5)
        .attr('y', -MARGIN.top + 5)
        .attr('width', COL_WIDTH)
        .attr('height', height - 10)
        .attr('fill', '#7c3aed22')
        .attr('rx', 4);
    }

    // Gates
    circuit_layout.columns.forEach((col) => {
      const x = col.column_index * COL_WIDTH + COL_WIDTH / 2;

      // Draw CNOT lines between ctrl and tgt
      const ctrlGates = col.gates.filter((g) => g.name === 'CNOT_ctrl');
      const tgtGates = col.gates.filter((g) => g.name === 'CNOT_tgt');
      ctrlGates.forEach((ctrl) => {
        const tgt = tgtGates.find((t) => t.step_index === ctrl.step_index);
        if (tgt) {
          const y1 = ctrl.qubit * QUBIT_HEIGHT + QUBIT_HEIGHT / 2;
          const y2 = tgt.qubit * QUBIT_HEIGHT + QUBIT_HEIGHT / 2;
          g.append('line')
            .attr('x1', x).attr('x2', x)
            .attr('y1', y1).attr('y2', y2)
            .attr('stroke', '#10b981').attr('stroke-width', 2);
        }
      });

      col.gates.forEach((gate) => {
        const y = gate.qubit * QUBIT_HEIGHT + QUBIT_HEIGHT / 2;
        const color = GATE_COLORS[gate.name] ?? GATE_COLORS.default;
        const isActive = gate.step_index === currentStep;

        if (gate.name === 'CNOT_ctrl') {
          g.append('circle')
            .attr('cx', x).attr('cy', y)
            .attr('r', 8)
            .attr('fill', color)
            .attr('stroke', isActive ? '#fff' : 'none')
            .attr('stroke-width', 1.5);
        } else if (gate.name === 'CNOT_tgt') {
          g.append('circle')
            .attr('cx', x).attr('cy', y)
            .attr('r', 12)
            .attr('fill', 'none')
            .attr('stroke', color)
            .attr('stroke-width', 2);
          g.append('line').attr('x1', x - 12).attr('x2', x + 12).attr('y1', y).attr('y2', y)
            .attr('stroke', color).attr('stroke-width', 2);
          g.append('line').attr('x1', x).attr('x2', x).attr('y1', y - 12).attr('y2', y + 12)
            .attr('stroke', color).attr('stroke-width', 2);
        } else {
          // Box gate — width adapts to label length
          const gw = GATE_W(gate.name);
          const displayName = GATE_LABEL(gate.name);
          g.append('rect')
            .attr('x', x - gw / 2).attr('y', y - GATE_SIZE / 2)
            .attr('width', gw).attr('height', GATE_SIZE)
            .attr('fill', color + (isActive ? 'ff' : '88'))
            .attr('stroke', isActive ? '#fff' : color)
            .attr('stroke-width', isActive ? 2 : 1)
            .attr('rx', 4);
          g.append('text')
            .attr('x', x).attr('y', y + 4)
            .attr('fill', '#fff').attr('font-size', 11).attr('font-weight', 600)
            .attr('text-anchor', 'middle')
            .text(displayName);
        }
      });
    });
  }, [trace, currentStep]);

  if (!trace) return null;

  const n_cols = trace.circuit_layout.columns.length;
  const n_qubits = trace.circuit_layout.qubit_labels.length;
  const svgWidth = MARGIN.left + n_cols * COL_WIDTH + MARGIN.right;

  return (
    <div className="card" ref={containerRef} style={{ overflowX: 'auto' }}>
      <div className="card-title">Circuit Diagram</div>
      <svg ref={svgRef} style={{ display: 'block', minWidth: svgWidth }} />
    </div>
  );
}
