from fastapi import APIRouter, HTTPException
from backend.models.requests import SimulateRequest, SweepRequest
from backend.models.responses import (
    SimulationTrace, AlgorithmsResponse, AlgorithmInfo, SweepResponse, SweepPoint
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
                         noise_config=request.noise_config)
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
