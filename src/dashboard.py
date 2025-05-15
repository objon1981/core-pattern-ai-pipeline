# FastAPI-based REST API

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from src.pipeline_manager import PipelineManager
from src.utils.dashboard_utils import load_environment

router = APIRouter()

# Load config once
with open("configs/pipeline_config.json") as f:
    CONFIG = json.load(f)

pipeline = PipelineManager(CONFIG)
env = load_environment(CONFIG)

class PipelineRequest(BaseModel):
    input_data: list

@router.post("/run_pipeline")
def run_pipeline(request: PipelineRequest):
    try:
        explanation = pipeline.run(request.input_data, env)
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
