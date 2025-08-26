import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging
import base64
import json
import httpx  # We'll use httpx for HTTP requests instead of openai client

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAT Tutor API", version="1.0.0")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Get API key
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

class ChatMessage(BaseModel):
    message: str

class APIResponse(BaseModel):
    response: str
    error: Optional[str] = None

# SAT Tutor system prompt
SAT_TUTOR_PROMPT = """You are an expert SAT tutor with years of experience helping students improve their scores.

Your approach:
1. Always provide the correct answer first
2. Give step-by-step explanations that are easy to follow
3. Identify the key concepts being tested
4. Point out common mistakes students make on similar problems
5. Provide study tips when relevant

Keep your explanations clear, encouraging, and educational. If you're unsure about a question, ask for clarification rather than guessing."""

async def call_openai_api(messages, model="gpt-4", max_tokens=800):
    """Direct API call to OpenAI using HTTP requests"""
    
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.3
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/chat/completions",
                headers=headers,
                json=payload
            )
            
            if response.status_code == 200:
                data = response.json()
                return data["choices"][0]["message"]["content"]
            else:
                error_data = response.json()
                error_msg = error_data.get("error", {}).get("message", "Unknown error")
                raise HTTPException(status_code=response.status_code, detail=f"OpenAI API error: {error_msg}")
                
        except httpx.TimeoutException:
            raise HTTPException(status_code=500, detail="Request to OpenAI timed out")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "SAT Tutor API is running successfully!",
        "status": "healthy",
        "openai_status": "✅ Connected (HTTP)" if OPENAI_API_KEY else "❌ API Key Missing",
        "environment": "railway" if os.getenv("RAILWAY_ENVIRONMENT") else "local",
        "api_key_present": bool(OPENAI_API_KEY),
        "api_key_preview": f"{OPENAI_API_KEY[:10]}..." if OPENAI_API_KEY else None,
        "method": "Direct HTTP requests (bypassing OpenAI client library)"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "openai_configured": bool(OPENAI_API_KEY),
        "method": "http_requests"
    }

@app.post("/api/analyze-question", response_model=APIResponse)
async def analyze_question(file: UploadFile = File(...)):
    """Analyze an uploaded SAT question image"""
    
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="OpenAI API key not configured. Please add OPENAI_API_KEY environment variable."
        )
    
    # Validate file type
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    # Check file size (limit to 10MB)
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
    
    try:
        logger.info(f"Processing image: {file.filename}")
        
        # Read and encode the image
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # Prepare messages for vision API
        messages = [
            {
                "role": "system",
                "content": SAT_TUTOR_PROMPT
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze this SAT practice question image. Provide the answer and explanation following your standard tutoring approach."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ]
        
        # Call OpenAI Vision API
        response_content = await call_openai_api(
            messages=messages, 
            model="gpt-4o", 
            max_tokens=1500
        )
        
        logger.info("Successfully analyzed image")
        return APIResponse(response=response_content)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/chat", response_model=APIResponse)
async def chat(message_data: ChatMessage):
    """Handle text-based chat messages"""
    
    if not OPENAI_API_KEY:
        raise HTTPException(
            status_code=500, 
            detail="OpenAI API key not configured. Please add OPENAI_API_KEY environment variable."
        )
    
    if not message_data.message or len(message_data.message.strip()) == 0:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if len(message_data.message) > 1000:
        raise HTTPException(status_code=400, detail="Message too long. Maximum 1000 characters")
    
    try:
        logger.info(f"Processing chat message: {message_data.message[:50]}...")
        
        # Prepare messages
        messages = [
            {"role": "system", "content": SAT_TUTOR_PROMPT},
            {"role": "user", "content": message_data.message}
        ]
        
        # Call OpenAI API
        response_content = await call_openai_api(messages=messages, max_tokens=800)
        
        logger.info("Successfully processed chat message")
        return APIResponse(response=response_content)
        
    except HTTPException:
        raise  # Re-raise HTTP exceptions as-is
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
