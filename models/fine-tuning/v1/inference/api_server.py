# api_server.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import asyncio
from typing import Dict, List, Any
from prometheus_client import start_http_server, exposition

# Import the core engine logic from the same directory
from inference_engine import MultiModelInferenceEngine, GenerationError, ModelLoadError

# Pydantic models for API request/response validation
class InferenceRequest(BaseModel):
    input_text: str
    generation_params: Dict[str, Any] = {}

class BatchInferenceRequest(BaseModel):
    inputs: List[str]
    generation_params: Dict[str, Any] = {}
    
class GenericDataPoint(BaseModel):
    data_type: str = Field(..., description="The type of data, e.g., 'ohlcv', 'news', 'onchain'.")
    data: Dict[str, Any] = Field(..., description="The raw data payload for inference.")

class GenericDataBatch(BaseModel):
    batch_id: str
    data_points: List[GenericDataPoint]

class APIEngine:
    """Manages the lifecycle and async calls to the MultiModelInferenceEngine."""
    def __init__(self):
        self.engine = MultiModelInferenceEngine()

    async def generate(self, model_name: str, input_text: str, **kwargs) -> Dict[str, Any]:
        return await asyncio.to_thread(self.engine.generate, model_name, input_text, **kwargs)

    async def batch_generate(self, model_name: str, inputs: List[str], **kwargs) -> List[Dict]:
        return await asyncio.to_thread(self.engine.batch_generate, model_name, inputs, **kwargs)
    
    async def process_data_batch(self, data_points: List[Dict[str, Any]]):
        return await asyncio.to_thread(self.engine.process_batch, data_points)

# Initialize the FastAPI app and the API engine
app = FastAPI(
    title="Multi-Model Inference API",
    description="A service for running inference on multiple small language models."
)
api_engine = APIEngine()

@app.on_event("startup")
async def startup_event():
    # Start Prometheus metrics server on a different port to avoid conflict with the FastAPI app
    start_http_server(8001)

@app.get("/")
def read_root():
    return {"message": "Multi-Model Inference API is running. Visit /docs for API documentation."}

@app.get("/models")
def list_available_models():
    """Lists all configured models and their details."""
    return api_engine.engine.list_models()

@app.get("/models/{model_name}/info")
def get_model_info(model_name: str):
    """Provides detailed information about a specific model."""
    info = api_engine.engine.get_model_info(model_name)
    if "error" in info:
        raise HTTPException(status_code=404, detail=info["error"])
    return info

@app.post("/generate/{model_name}")
async def generate_text(model_name: str, request: InferenceRequest):
    """Runs a single inference request on the specified model."""
    try:
        result = await api_engine.generate(
            model_name=model_name,
            input_text=request.input_text,
            **request.generation_params
        )
        return result
    except (ModelLoadError, GenerationError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/batch_generate/{model_name}")
async def batch_generate_text(model_name: str, request: BatchInferenceRequest):
    """Runs a batch of inference requests on the specified model."""
    try:
        results = await api_engine.batch_generate(
            model_name=model_name,
            inputs=request.inputs,
            **request.generation_params
        )
        return results
    except (ModelLoadError, GenerationError) as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {e}")

@app.post("/process_data")
async def process_data_batch(request: GenericDataBatch):
    """
    Processes a generic batch of data points, routing them to the correct models
    based on the 'data_type' field.
    """
    try:
        data_dicts = [dp.data for dp in request.data_points]
        results = await api_engine.process_data_batch(data_dicts)
        return {"batch_id": request.batch_id, "results": results}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Data batch processing failed: {e}")