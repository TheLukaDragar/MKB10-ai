import os
import openai
import logging
from datetime import datetime, timedelta
from colorama import Fore, Style
import dotenv
from fastapi import FastAPI, BackgroundTasks
import uvicorn
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional
import asyncio
import json
import atexit
import httpx
import time

dotenv.load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI()

# Worker registration file
WORKERS_FILE = "available_workers.json"

def register_worker(port: int, worker_id: int):
    """Register this worker in the available workers file"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            workers = []
            if os.path.exists(WORKERS_FILE):
                try:
                    with open(WORKERS_FILE, 'r') as f:
                        workers = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Corrupted workers file, resetting... (attempt {attempt + 1}/{max_retries})")
                    workers = []
            
            # Remove any existing entry for this worker and any dead workers
            current_time = datetime.now()
            workers = [w for w in workers if 
                      (w['worker_id'] != worker_id and 
                       (current_time - datetime.fromisoformat(w['start_time'])) <= timedelta(minutes=5))]
            
            # Add this worker
            workers.append({
                'worker_id': worker_id,
                'port': port,
                'start_time': datetime.now().isoformat(),
                'host': 'localhost'
            })
            
            # Write back to file with file locking
            temp_file = f"{WORKERS_FILE}.tmp"
            with open(temp_file, 'w') as f:
                json.dump(workers, f, indent=2)
            os.replace(temp_file, WORKERS_FILE)  # Atomic operation
            
            logger.info(f"Successfully registered worker {worker_id} on port {port}")
            return
        except Exception as e:
            logger.error(f"Error registering worker (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

def unregister_worker(worker_id: int):
    """Remove this worker from the available workers file"""
    max_retries = 3
    retry_delay = 1  # seconds
    
    for attempt in range(max_retries):
        try:
            if os.path.exists(WORKERS_FILE):
                workers = []
                try:
                    with open(WORKERS_FILE, 'r') as f:
                        workers = json.load(f)
                except json.JSONDecodeError:
                    logger.warning(f"Corrupted workers file during unregister, resetting... (attempt {attempt + 1}/{max_retries})")
                    workers = []
                
                # Remove this worker
                workers = [w for w in workers if w['worker_id'] != worker_id]
                
                # Write back to file with file locking
                temp_file = f"{WORKERS_FILE}.tmp"
                with open(temp_file, 'w') as f:
                    json.dump(workers, f, indent=2)
                os.replace(temp_file, WORKERS_FILE)  # Atomic operation
                
                logger.info(f"Successfully unregistered worker {worker_id}")
                return
        except Exception as e:
            logger.error(f"Error unregistering worker (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)

# OpenAI client setup with httpx
async def get_openai_client():
    return openai.AsyncOpenAI(
        api_key=os.getenv("OPENAI_API_KEY"),
        base_url="http://localhost:8001/v1",
        http_client=httpx.AsyncClient()
    )

class LLMRequest(BaseModel):
    prompt: str
    model_name: str = "nemotron"
    max_tokens: int = 4000
    worker_id: int
    extra_kwargs: Optional[Dict[str, Any]] = Field(default_factory=dict)

async def llm_call(prompt: str, model_name: str = "nemotron", worker_id: int = 0, **kwargs):
    """
    Async function to handle LLM calls with streaming output
    """
    logger.info(f"Worker {worker_id} - {'='*80}")
    logger.info(f"Worker {worker_id} - 🤖 Starting LLM call with model: {model_name}")
    logger.info(f"Worker {worker_id} - {'-'*40} PROMPT {'-'*40}")
    logger.info(f"Worker {worker_id} - \033[94m{prompt}\033[0m")
    logger.info(f"Worker {worker_id} - {'-'*80}")
    
    client = await get_openai_client()
    
    # Merge all kwargs for the API call
    api_kwargs = {
        "model": model_name,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": kwargs.pop('max_tokens', 4000),
        "stream": True,
        **kwargs  # Include any remaining kwargs
    }
    
    full_response = ""
    start_time = datetime.now()
    
    logger.info(f"Worker {worker_id} - {'-'*40} RESPONSE {'-'*39}")
    
    try:
        stream = await client.chat.completions.create(**api_kwargs)
        async for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                content = chunk.choices[0].delta.content
                full_response += content
                # Print chunks in green with no newline
                print(f"\033[32m{content}\033[0m", end="", flush=True)
    except Exception as e:
        logger.error(f"Error in streaming: {e}")
        raise
    finally:
        await client.close()
    
    duration = (datetime.now() - start_time).total_seconds()
    print()  # Add newline after streaming response
    logger.info(f"Worker {worker_id} - {'-'*80}")
    logger.info(f"Worker {worker_id} - ✅ LLM call completed in {duration:.2f} seconds")
    logger.info(f"Worker {worker_id} - {'='*80}")
    
    return full_response

@app.post("/process")
async def process_request(request: LLMRequest):
    """
    Endpoint to handle LLM requests
    """
    # Combine all kwargs
    kwargs = {
        "model_name": request.model_name,
        "max_tokens": request.max_tokens,
        "worker_id": request.worker_id,
        **request.extra_kwargs  # Include any additional kwargs
    }
    
    result = await llm_call(
        prompt=request.prompt,
        **kwargs
    )
    return {"result": result}

@app.on_event("startup")
async def startup_event():
    """Register worker on startup"""
    if 'current_worker_id' in globals() and 'current_port' in globals():
        register_worker(current_port, current_worker_id)

@app.on_event("shutdown")
async def shutdown_event():
    """Unregister worker on shutdown"""
    if 'current_worker_id' in globals():
        unregister_worker(current_worker_id)

def start_worker(port: int, worker_id: int):
    """
    Start the worker server on the specified port
    """
    # Store worker info globally for startup/shutdown events
    global current_worker_id, current_port
    current_worker_id = worker_id
    current_port = port
    
    print(f"{Fore.CYAN}Starting worker {worker_id} on port {port}{Style.RESET_ALL}")
    
    # Register worker
    register_worker(port, worker_id)
    
    # Register cleanup on exit
    atexit.register(lambda: unregister_worker(worker_id))
    
    # Start the server
    uvicorn.run(app, host="0.0.0.0", port=port)

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "worker_id": current_worker_id}

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, required=True, help="Port to run the worker on")
    parser.add_argument("--worker-id", type=int, required=True, help="Worker ID")
    args = parser.parse_args()
    
    start_worker(args.port, args.worker_id) 