import asyncio
import json
import logging
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .vector_store import AsyncChromaManager
from .graph_store import AsyncGraphManager
from .exceptions import InfrastructureDegradationError, DatabaseConnectionError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Symantix SRE Diagnostics API", version="1.0.0")

# Security Middleware: CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Security: Basic API Key Validation
from .config import settings
API_KEY = settings.api_key

async def verify_api_key(request: Request):
    key = request.headers.get("X-API-Key")
    if key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API Key"
        )
    return key

# Initialize global managers
chroma_manager = AsyncChromaManager()
graph_manager = AsyncGraphManager()

class DiagnosticRequest(BaseModel):
    """Strict Pydantic payload validation for the diagnosis route."""
    incident_description: str = Field(..., description="Natural language description of the SRE incident.")
    max_hops: int = Field(2, description="Number of hops for Neo4j topological traversal.")

@app.on_event("startup")
async def startup_event():
    """Execute required startup indexes and connections."""
    logger.info("Initializing Neo4j Graph Indexes...")
    await graph_manager.initialize_schema()
    logger.info("Initializing ChromaDB Collections...")
    await chroma_manager.initialize_collection()

async def mock_vllm_stream(prompt: str):
    """
    Simulates the vLLM/Ollama inference engine stream.
    In a true production deployment, this uses an async HTTP client (like httpx) 
    to connect to the local vLLM OpenAI-compatible endpoint.
    """
    # Deterministic JSON Root Cause Analysis output
    simulated_response = {
        "root_cause": "Semantic overlap detected in Service dependency graphs matching known database exhaustion signatures.",
        "blast_radius": ["Frontend-Gateway", "Auth-Service"],
        "remediation": "Scale the PostgreSQL connection pool limit and temporarily rate limit the Frontend-Gateway."
    }
    
    json_str = json.dumps(simulated_response)
    
    # Stream character by character to simulate token generation via SSE
    try:
        for i in range(0, len(json_str), 5):
            chunk = json_str[i:i+5]
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.02)
        yield "data: [DONE]\n\n"
    except asyncio.CancelledError:
        logger.warning("Client disconnected prematurely during SSE generation.")
        raise

@app.post("/v1/diagnose", dependencies=[Depends(verify_api_key)])
async def diagnose_incident(request: DiagnosticRequest):
    """
    Dual-Query Router:
    1. Semantic Retrieval (ChromaDB)
    2. Topological Traversal (Neo4j)
    3. Context Aggregation & SLM Inference
    """
    query = request.incident_description
    
    try:
        # 1. Semantic Vector Lookup
        logger.info(f"Querying ChromaDB for semantics: {query}")
        try:
            vector_results = await chroma_manager.retrieve_similar(query, top_k=5)
            # Extract trace_ids from ChromaDB hits
            trace_ids = []
            if vector_results and "metadatas" in vector_results and vector_results["metadatas"][0]:
                trace_ids = [meta.get("trace_id") for meta in vector_results["metadatas"][0] if meta.get("trace_id")]
            context_buffer = [f"Vector Context: {doc}" for doc in vector_results.get("documents", [[]])[0]]
        except DatabaseConnectionError as e:
            logger.warning(f"Graceful Degradation triggered for Vector Store: {e}")
            trace_ids = []
            context_buffer = ["Vector Context: [UNAVAILABLE DUE TO CHROMADB OUTAGE]"]
            
        # 2. Graph Topological Traversal
        if trace_ids:
            try:
                logger.info(f"Querying Neo4j for topology on trace_ids: {trace_ids}")
                graph_results = await graph_manager.traverse_topology(trace_ids, hops=request.max_hops)
                for record in graph_results:
                    context_buffer.append(f"Graph Topology: {record['root_service']} -> {record['dependency_chain']}")
            except InfrastructureDegradationError as e:
                logger.warning(f"Graceful Degradation triggered: {e}")
                context_buffer.append("Graph Topology: [UNAVAILABLE DUE TO NEO4J OUTAGE]")
        
        aggregated_context = "\n".join(context_buffer)
        
        # 3. Local Model Serving Engine (vLLM/Ollama via streaming)
        system_prompt = f"Analyze the following system log trace context and determine the primary root cause, failure blast radius, and recovery steps.\n\nContext:\n{aggregated_context}"
        
        return StreamingResponse(mock_vllm_stream(system_prompt), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Failed to process diagnostic request: {e}")
        raise HTTPException(status_code=500, detail=str(e))
