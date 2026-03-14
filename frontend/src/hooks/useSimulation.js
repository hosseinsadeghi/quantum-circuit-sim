import { useEffect } from 'react';
import { useSimulationContext } from '../context/SimulationContext.jsx';

export function useSimulation() {
  const ctx = useSimulationContext();

  useEffect(() => {
    ctx.loadAlgorithms();
  }, []);

  return ctx;
}
