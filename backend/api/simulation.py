import asyncio
import uuid
from fastapi import APIRouter, HTTPException, WebSocket, WebSocketDisconnect
from backend.models.requests import SimulateRequest, SweepRequest, AsyncSimulateRequest
from backend.models.responses import (
    SimulationTrace, AlgorithmsResponse, AlgorithmInfo, SweepResponse, SweepPoint,
    JobSubmittedResponse, JobStatusResponse,
)
from algorithms.bell_state import BellStateAlgorithm
from algorithms.grover import GroverAlgorithm
from algorithms.qaoa_maxcut import QAOAMaxCutAlgorithm
from algorithms.deutsch_jozsa import DeutschJozsaAlgorithm
from algorithms.bernstein_vazirani import BernsteinVaziraniAlgorithm
from algorithms.qft import QFTAlgorithm
from algorithms.phase_estimation import PhaseEstimationAlgorithm
from algorithms.ghz import GHZAlgorithm
from algorithms.teleportation import TeleportationAlgorithm
from algorithms.ising_evolution import IsingEvolutionAlgorithm
from algorithms.rabi import RabiAlgorithm
from algorithms.vqe import VQEAlgorithm
from algorithms.error_correction import ErrorCorrectionAlgorithm
from algorithms.ma_qaoa import MAQAOAAlgorithm
from algorithms.adapt_qaoa import ADAPTQAOAAlgorithm
import json

router = APIRouter()

ALGORITHMS = {
    alg.algorithm_id: alg
    for alg in [
        BellStateAlgorithm(),
        GHZAlgorithm(),
        TeleportationAlgorithm(),
        GroverAlgorithm(),
        DeutschJozsaAlgorithm(),
        BernsteinVaziraniAlgorithm(),
        QFTAlgorithm(),
        PhaseEstimationAlgorithm(),
        QAOAMaxCutAlgorithm(),
        VQEAlgorithm(),
        IsingEvolutionAlgorithm(),
        RabiAlgorithm(),
        ErrorCorrectionAlgorithm(),
        MAQAOAAlgorithm(),
        ADAPTQAOAAlgorithm(),
    ]
}


@router.get("/algorithms", response_model=AlgorithmsResponse)
async def get_algorithms():
    return AlgorithmsResponse(
        algorithms=[
            AlgorithmInfo(
                algorithm_id=alg.algorithm_id,
                name=alg.name,
                description=alg.description,
                parameter_schema=alg.parameter_schema,
                category=getattr(alg, "category", "general"),
            )
            for alg in ALGORITHMS.values()
        ]
    )


@router.post("/simulate", response_model=SimulationTrace)
async def simulate(request: SimulateRequest):
    alg = ALGORITHMS.get(request.algorithm_id)
    if alg is None:
        raise HTTPException(status_code=404, detail=f"Algorithm '{request.algorithm_id}' not found")

    try:
        result = alg.run(request.parameters, mode=request.mode,
                         noise_config=request.noise_config,
                         optimize=request.optimize)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    try:
        json.dumps(result)
    except TypeError as e:
        raise HTTPException(status_code=500, detail=f"Serialization error: {e}")

    return result


@router.post("/sweep", response_model=SweepResponse)
async def sweep(request: SweepRequest):
    """Run an algorithm across a range of parameter values and return final probabilities."""
    alg = ALGORITHMS.get(request.algorithm_id)
    if alg is None:
        raise HTTPException(status_code=404, detail=f"Algorithm '{request.algorithm_id}' not found")

    points: list[SweepPoint] = []
    for value in request.sweep_values:
        params = {**request.fixed_parameters, request.sweep_parameter: value}
        try:
            result = alg.run(params, mode=request.mode, noise_config=request.noise_config)
        except (ValueError, KeyError) as e:
            raise HTTPException(status_code=400, detail=f"sweep value {value!r}: {e}")

        meas = result["measurement"]
        points.append(SweepPoint(
            parameter_value=value,
            most_likely_outcome=meas["most_likely_outcome"],
            probabilities=meas["probabilities"],
            basis_labels=meas["basis_labels"],
        ))

    return SweepResponse(
        algorithm=request.algorithm_id,
        sweep_parameter=request.sweep_parameter,
        points=points,
    )


# ---------------------------------------------------------------------------
# Async execution with job polling (Tier 4.1)
# ---------------------------------------------------------------------------

# In-memory job store (production would use Redis/DB)
_jobs: dict[str, dict] = {}


def _run_simulation_sync(algorithm_id: str, parameters: dict, mode: str,
                         noise_config, optimize: bool) -> dict:
    """Run simulation synchronously (called in thread)."""
    alg = ALGORITHMS[algorithm_id]
    return alg.run(parameters, mode=mode, noise_config=noise_config, optimize=optimize)


@router.post("/simulate/async", response_model=JobSubmittedResponse)
async def simulate_async(request: AsyncSimulateRequest):
    """Submit a simulation job for async execution. Returns a job_id for polling."""
    alg = ALGORITHMS.get(request.algorithm_id)
    if alg is None:
        raise HTTPException(status_code=404, detail=f"Algorithm '{request.algorithm_id}' not found")

    job_id = str(uuid.uuid4())
    _jobs[job_id] = {"status": "pending", "result": None, "error": None}

    async def _run_job():
        _jobs[job_id]["status"] = "running"
        try:
            result = await asyncio.to_thread(
                _run_simulation_sync,
                request.algorithm_id,
                request.parameters,
                request.mode,
                request.noise_config,
                request.optimize,
            )
            _jobs[job_id]["status"] = "completed"
            _jobs[job_id]["result"] = result
        except Exception as e:
            _jobs[job_id]["status"] = "failed"
            _jobs[job_id]["error"] = str(e)

    asyncio.create_task(_run_job())
    return JobSubmittedResponse(job_id=job_id, status="pending")


@router.get("/simulate/job/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """Poll for async simulation job status."""
    if job_id not in _jobs:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")

    job = _jobs[job_id]
    return JobStatusResponse(
        job_id=job_id,
        status=job["status"],
        result=job["result"],
        error=job["error"],
    )


# ---------------------------------------------------------------------------
# WebSocket streaming (Tier 4.2)
# ---------------------------------------------------------------------------

@router.websocket("/simulate/stream")
async def simulate_stream(websocket: WebSocket):
    """Stream simulation steps over WebSocket as they're computed.

    Client sends a JSON SimulateRequest, server streams back individual steps.
    """
    await websocket.accept()
    try:
        data = await websocket.receive_json()
        algorithm_id = data.get("algorithm_id")
        parameters = data.get("parameters", {})
        mode = data.get("mode", "statevector")
        noise_config = data.get("noise_config")
        optimize = data.get("optimize", False)

        alg = ALGORITHMS.get(algorithm_id)
        if alg is None:
            await websocket.send_json({"error": f"Algorithm '{algorithm_id}' not found"})
            await websocket.close()
            return

        # Run in thread to not block event loop
        result = await asyncio.to_thread(
            _run_simulation_sync,
            algorithm_id, parameters, mode, noise_config, optimize,
        )

        # Stream steps one by one
        for step in result["steps"]:
            await websocket.send_json({"type": "step", "data": step})

        # Send final measurement and layout
        await websocket.send_json({
            "type": "complete",
            "measurement": result["measurement"],
            "circuit_layout": result["circuit_layout"],
            "n_qubits": result["n_qubits"],
        })

    except WebSocketDisconnect:
        pass
    except Exception as e:
        try:
            await websocket.send_json({"error": str(e)})
        except Exception:
            pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
