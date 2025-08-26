import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import logging

# Only import openai if we can, to avoid import errors during deployment
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

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

# Initialize OpenAI client only if available and API key exists
openai_client = None
openai_error = None

if OPENAI_AVAILABLE:
    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        try:
            # Simple client initialization without any extra parameters
            openai_client = openai.OpenAI(
                api_key=api_key,
                timeout=30.0  # Only add timeout, which is always supported
            )
            logger.info("OpenAI client initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAI client: {e}")
            openai_error = f"Client initialization failed: {str(e)}"
            
            # Try even simpler initialization as fallback
            try:
                openai_client = openai.OpenAI()  # Let it use env var automatically
                logger.info("OpenAI client initialized with env var fallback")
                openai_error = None
            except Exception as e2:
                logger.error(f"Fallback initialization also failed: {e2}")
                openai_error = f"Both init attempts failed: {str(e)} | {str(e2)}"
    else:
        openai_error = "API key not found"
else:
    openai_error = "OpenAI package not available"

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

@app.get("/")
async def root():
    # Debug information
    api_key = os.getenv("OPENAI_API_KEY")
    api_key_preview = f"{api_key[:10]}..." if api_key else None
    
    openai_status = "❌ Not Available"
    if not OPENAI_AVAILABLE:
        openai_status = "❌ OpenAI package not installed"
    elif not api_key:
        openai_status = "❌ API Key Missing"
    elif openai_client:
        openai_status = "✅ Connected"
    else:
        openai_status = f"❌ Connection Failed: {openai_error}"
    
    return {
        "message": "SAT Tutor API is running successfully!",
        "status": "healthy",
        "openai_status": openai_status,
        "environment": "railway" if os.getenv("RAILWAY_ENVIRONMENT") else "local",
        "api_key_present": bool(api_key),
        "api_key_preview": api_key_preview,
        "openai_available": OPENAI_AVAILABLE,
        "client_created": bool(openai_client),
        "error_details": openai_error
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "openai_configured": bool(openai_client),
        "openai_available": OPENAI_AVAILABLE
    }

@app.post("/api/analyze-question", response_model=APIResponse)
async def analyze_question(file: UploadFile = File(...)):
    """Analyze an uploaded SAT question image"""
    
    if not OPENAI_AVAILABLE:
        raise HTTPException(
            status_code=500, 
            detail="OpenAI package not available. Please check deployment."
        )
    
    if not openai_client:
        raise HTTPException(
            status_code=500, 
            detail="OpenAI not configured. Please add OPENAI_API_KEY environment variable."
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
        import base64
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # Call OpenAI Vision API
        response = openai_client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "system",
                "content": SAT_TUTOR_PROMPT
            }, {
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
            }],
            max_tokens=1500,
            temperature=0.3
        )
        
        logger.info("Successfully analyzed image")
        return APIResponse(response=response.choices[0].message.content)
        
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        error_message = "Internal server error"
        
        if "insufficient_quota" in str(e).lower():
            error_message = "OpenAI API quota exceeded. Please check your billing."
        elif "invalid_api_key" in str(e).lower():
            error_message = "Invalid OpenAI API key."
        
        raise HTTPException(status_code=500, detail=error_message)

@app.post("/api/chat", response_model=APIResponse)
async def chat(message_data: ChatMessage):
    """Handle text-based chat messages"""
    
    if not OPENAI_AVAILABLE:
        raise HTTPException(
            status_code=500, 
            detail="OpenAI package not available. Please check deployment."
        )
    
    if not openai_client:
        raise HTTPException(
            status_code=500, 
            detail="OpenAI not configured. Please add OPENAI_API_KEY environment variable."
        )
    
    if not message_data.message or len(message_data.message.strip()) == 0:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if len(message_data.message) > 1000:
        raise HTTPException(status_code=400, detail="Message too long. Maximum 1000 characters")
    
    try:
        logger.info(f"Processing chat message: {message_data.message[:50]}...")
        
        response = openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SAT_TUTOR_PROMPT},
                {"role": "user", "content": message_data.message}
            ],
            max_tokens=800,
            temperature=0.3
        )
        
        logger.info("Successfully processed chat message")
        return APIResponse(response=response.choices[0].message.content)
        
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        error_message = "Internal server error"
        
        if "insufficient_quota" in str(e).lower():
            error_message = "OpenAI API quota exceeded. Please check your billing."
        elif "invalid_api_key" in str(e).lower():
            error_message = "Invalid OpenAI API key."
        
        raise HTTPException(status_code=500, detail=error_message)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
