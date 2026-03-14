from fastapi import APIRouter
from backend.api.simulation import router as simulation_router
from backend.models.responses import HealthResponse

router = APIRouter()
router.include_router(simulation_router, prefix="/api")


@router.get("/api/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="ok", version="0.1.0")
