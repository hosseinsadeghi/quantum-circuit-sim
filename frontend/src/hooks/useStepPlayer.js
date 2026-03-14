import { useState, useEffect, useRef, useCallback } from 'react';

export function useStepPlayer(totalSteps, currentStep, setCurrentStep) {
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(800); // ms per step
  const intervalRef = useRef(null);

  const stop = useCallback(() => {
    setPlaying(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
  }, []);

  const play = useCallback(() => {
    if (currentStep >= totalSteps - 1) {
      setCurrentStep(0);
    }
    setPlaying(true);
  }, [currentStep, totalSteps, setCurrentStep]);

  const stepForward = useCallback(() => {
    stop();
    setCurrentStep((s) => Math.min(s + 1, totalSteps - 1));
  }, [stop, totalSteps, setCurrentStep]);

  const stepBackward = useCallback(() => {
    stop();
    setCurrentStep((s) => Math.max(s - 1, 0));
  }, [stop, setCurrentStep]);

  useEffect(() => {
    if (playing) {
      intervalRef.current = setInterval(() => {
        setCurrentStep((s) => {
          if (s >= totalSteps - 1) {
            setPlaying(false);
            clearInterval(intervalRef.current);
            return s;
          }
          return s + 1;
        });
      }, speed);
    }
    return () => clearInterval(intervalRef.current);
  }, [playing, speed, totalSteps, setCurrentStep]);

  // Keyboard shortcuts
  useEffect(() => {
    const handler = (e) => {
      if (e.target.tagName === 'INPUT') return;
      if (e.code === 'Space') { e.preventDefault(); playing ? stop() : play(); }
      else if (e.code === 'ArrowRight') stepForward();
      else if (e.code === 'ArrowLeft') stepBackward();
    };
    window.addEventListener('keydown', handler);
    return () => window.removeEventListener('keydown', handler);
  }, [playing, play, stop, stepForward, stepBackward]);

  return { playing, play, stop, stepForward, stepBackward, speed, setSpeed };
}
