import { useEffect } from 'react';
import './App.css';
import { SimulationProvider, useSimulationContext } from './context/SimulationContext.jsx';
import AlgorithmSelector from './components/AlgorithmSelector.jsx';
import ParameterPanel from './components/ParameterPanel.jsx';
import StepScrubber from './components/StepScrubber.jsx';
import CircuitDiagram from './components/CircuitDiagram.jsx';
import AmplitudeChart from './components/AmplitudeChart.jsx';
import MeasurementHistogram from './components/MeasurementHistogram.jsx';
import PhaseWheel from './components/PhaseWheel.jsx';

function AppContent() {
  const { error, trace, showPhaseWheel, setShowPhaseWheel, loadAlgorithms } = useSimulationContext();

  useEffect(() => {
    loadAlgorithms();
  }, []);

  return (
    <div className="app">
      <header className="app-header">
        <h1>Quantum Algorithm Explorer</h1>
        <p>Step-by-step quantum circuit simulation with interactive visualizations</p>
        {trace && (
          <label style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '6px', fontSize: '0.85rem', cursor: 'pointer' }}>
            <input type="checkbox" checked={showPhaseWheel} onChange={(e) => setShowPhaseWheel(e.target.checked)} />
            Show Phase Wheels
          </label>
        )}
      </header>

      <aside className="sidebar">
        <AlgorithmSelector />
        <ParameterPanel />
      </aside>

      <main className="main-content">
        {error && <div className="error-box">{error}</div>}
        {!trace && !error && (
          <div className="loading">Select an algorithm and click Run Simulation to begin.</div>
        )}
        {trace && (
          <>
            <StepScrubber />
            <CircuitDiagram />
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '16px' }}>
              <AmplitudeChart />
              <MeasurementHistogram />
            </div>
            <PhaseWheel />
          </>
        )}
      </main>

      <footer className="app-footer">
        <span style={{ fontSize: '0.8rem', color: '#4a5568' }}>
          Quantum Algorithm Explorer v0.1.0 — Pure NumPy simulation
        </span>
      </footer>
    </div>
  );
}

export default function App() {
  return (
    <SimulationProvider>
      <AppContent />
    </SimulationProvider>
  );
}
