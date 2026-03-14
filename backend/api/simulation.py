from fastapi import APIRouter, HTTPException
from backend.models.requests import SimulateRequest
from backend.models.responses import SimulationTrace, AlgorithmsResponse, AlgorithmInfo
from algorithms.bell_state import BellStateAlgorithm
from algorithms.grover import GroverAlgorithm
from algorithms.qaoa_maxcut import QAOAMaxCutAlgorithm
import json

router = APIRouter()

ALGORITHMS = {
    alg.algorithm_id: alg
    for alg in [BellStateAlgorithm(), GroverAlgorithm(), QAOAMaxCutAlgorithm()]
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
        result = alg.run(request.parameters)
    except (ValueError, KeyError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Validate JSON serializability (catches NumPy type leaks)
    try:
        json.dumps(result)
    except TypeError as e:
        raise HTTPException(status_code=500, detail=f"Serialization error: {e}")

    return result
