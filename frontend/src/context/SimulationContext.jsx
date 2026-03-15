import { createContext, useContext, useState, useCallback } from 'react';
import { fetchAlgorithms, runSimulation } from '../api/client.js';

const SimulationContext = createContext(null);

export function SimulationProvider({ children }) {
  const [algorithms, setAlgorithms] = useState([]);
  const [selectedAlgorithm, setSelectedAlgorithm] = useState(null);
  const [parameters, setParameters] = useState({});
  const [trace, setTrace] = useState(null);
  const [currentStep, setCurrentStep] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showPhaseWheel, setShowPhaseWheel] = useState(false);
  const [showObservables, setShowObservables] = useState(false);
  const [mode, setMode] = useState('statevector');
  const [noiseConfig, setNoiseConfig] = useState(null);

  const loadAlgorithms = useCallback(async () => {
    try {
      const data = await fetchAlgorithms();
      setAlgorithms(data.algorithms);
      if (data.algorithms.length > 0) {
        setSelectedAlgorithm(data.algorithms[0]);
        setParameters(getDefaultParameters(data.algorithms[0].parameter_schema));
      }
    } catch (e) {
      setError(e.message);
    }
  }, []);

  const selectAlgorithm = useCallback((alg) => {
    setSelectedAlgorithm(alg);
    setParameters(getDefaultParameters(alg.parameter_schema));
    setTrace(null);
    setCurrentStep(0);
    setError(null);
  }, []);

  const simulate = useCallback(async () => {
    if (!selectedAlgorithm) return;
    setLoading(true);
    setError(null);
    try {
      const result = await runSimulation(
        selectedAlgorithm.algorithm_id,
        parameters,
        mode,
        noiseConfig,
      );
      setTrace(result);
      setCurrentStep(0);
    } catch (e) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  }, [selectedAlgorithm, parameters, mode, noiseConfig]);

  const value = {
    algorithms,
    selectedAlgorithm,
    parameters,
    setParameters,
    trace,
    currentStep,
    setCurrentStep,
    loading,
    error,
    showPhaseWheel,
    setShowPhaseWheel,
    showObservables,
    setShowObservables,
    mode,
    setMode,
    noiseConfig,
    setNoiseConfig,
    loadAlgorithms,
    selectAlgorithm,
    simulate,
  };

  return <SimulationContext.Provider value={value}>{children}</SimulationContext.Provider>;
}

export function useSimulationContext() {
  const ctx = useContext(SimulationContext);
  if (!ctx) throw new Error('useSimulationContext must be used inside SimulationProvider');
  return ctx;
}

function getDefaultParameters(schema) {
  const defaults = {};
  if (!schema?.properties) return defaults;
  for (const [key, def] of Object.entries(schema.properties)) {
    if (def.default !== undefined) defaults[key] = def.default;
    else if (def.type === 'integer') defaults[key] = def.minimum ?? 1;
    else if (def.type === 'number') defaults[key] = def.minimum ?? 0;
    else if (def.type === 'string' && def.enum) defaults[key] = def.enum[0];
    else if (def.type === 'string') defaults[key] = '';
  }
  return defaults;
}
