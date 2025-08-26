# Updated main.py for Railway with Knowledge Base Integration

import os
import httpx
import json
import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import base64

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAT Tutor API with Knowledge Base", version="2.0.0")

# CORS configuration
ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Get configuration
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
ASSISTANT_ID = os.getenv("ASSISTANT_ID")

class ChatMessage(BaseModel):
    message: str

class APIResponse(BaseModel):
    response: str
    error: Optional[str] = None

async def call_openai_assistants_api(message_content, image_data=None):
    """Call OpenAI Assistants API with knowledge base"""
    
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    if not ASSISTANT_ID:
        raise HTTPException(status_code=500, detail="Assistant not configured. Please set up your knowledge base first.")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2"
    }
    
    async with httpx.AsyncClient(timeout=60.0) as client:
        try:
            # Step 1: Create a thread
            thread_response = await client.post(
                "https://api.openai.com/v1/threads",
                headers=headers,
                json={}
            )
            
            if thread_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to create thread")
            
            thread_id = thread_response.json()["id"]
            
            # Step 2: Add message to thread
            message_data = {
                "role": "user",
                "content": message_content if isinstance(message_content, str) else message_content
            }
            
            message_response = await client.post(
                f"https://api.openai.com/v1/threads/{thread_id}/messages",
                headers=headers,
                json=message_data
            )
            
            if message_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to add message to thread")
            
            # Step 3: Run the assistant
            run_data = {
                "assistant_id": ASSISTANT_ID,
                "instructions": """You are Bucky, a math tutor working for Caddell Prep. Follow these steps for every question:

1. IDENTIFY the question type (e.g., "This is a system of equations problem")
2. SEARCH your knowledge base for similar examples and Caddell Prep methods
3. EXPLAIN the approach step-by-step using our teaching style
4. PROVIDE the correct answer first, then detailed explanation
5. If applicable, mention Desmos calculator usage: https://www.desmos.com/calculator
6. Look for Caddell Prep tricks: answer choice testing, plugging in numbers, graphing
7. Use proper math formatting: $x^2$ for inline, $equations$ for display
8. Be friendly and instructional, like tutoring a confused student

Always base explanations on the uploaded Caddell Prep materials."""
            }
            
            run_response = await client.post(
                f"https://api.openai.com/v1/threads/{thread_id}/runs",
                headers=headers,
                json=run_data
            )
            
            if run_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to start assistant run")
            
            run_id = run_response.json()["id"]
            
            # Step 4: Wait for completion
            max_attempts = 60  # 60 seconds timeout
            attempts = 0
            
            while attempts < max_attempts:
                status_response = await client.get(
                    f"https://api.openai.com/v1/threads/{thread_id}/runs/{run_id}",
                    headers=headers
                )
                
                if status_response.status_code != 200:
                    raise HTTPException(status_code=500, detail="Failed to check run status")
                
                run_status = status_response.json()["status"]
                
                if run_status == "completed":
                    break
                elif run_status in ["failed", "cancelled", "expired"]:
                    error_msg = status_response.json().get("last_error", {}).get("message", "Unknown error")
                    raise HTTPException(status_code=500, detail=f"Assistant run failed: {error_msg}")
                
                await asyncio.sleep(1)
                attempts += 1
            
            if attempts >= max_attempts:
                raise HTTPException(status_code=500, detail="Assistant response timed out")
            
            # Step 5: Get the response
            messages_response = await client.get(
                f"https://api.openai.com/v1/threads/{thread_id}/messages",
                headers=headers
            )
            
            if messages_response.status_code != 200:
                raise HTTPException(status_code=500, detail="Failed to retrieve messages")
            
            messages = messages_response.json()["data"]
            assistant_message = next((msg for msg in messages if msg["role"] == "assistant"), None)
            
            if not assistant_message:
                raise HTTPException(status_code=500, detail="No assistant response found")
            
            return assistant_message["content"][0]["text"]["value"]
            
        except httpx.TimeoutException:
            raise HTTPException(status_code=500, detail="Request to OpenAI timed out")
        except Exception as e:
            logger.error(f"Assistant API error: {str(e)}")
            raise HTTPException(status_code=500, detail=f"API request failed: {str(e)}")

@app.get("/")
async def root():
    return {
        "message": "SAT Tutor API with Caddell Prep Knowledge Base",
        "status": "healthy",
        "openai_status": "✅ Connected (HTTP)" if OPENAI_API_KEY else "❌ API Key Missing",
        "knowledge_base": "✅ Ready" if ASSISTANT_ID else "❌ Not Configured",
        "environment": "railway" if os.getenv("RAILWAY_ENVIRONMENT") else "local",
        "version": "2.0.0 - Knowledge Base Enabled"
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy", 
        "openai_configured": bool(OPENAI_API_KEY),
        "assistant_configured": bool(ASSISTANT_ID),
        "knowledge_base": "enabled"
    }

@app.post("/api/setup-knowledge-base")
async def setup_knowledge_base():
    """Create assistant with knowledge base - run this once after uploading PDFs"""
    
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key required")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
        "OpenAI-Beta": "assistants=v2"
    }
    
    assistant_data = {
        "name": "Bucky - Caddell Prep SAT Tutor",
        "instructions": """You are Bucky, Caddell Prep's expert SAT tutor with access to comprehensive lesson materials, question types, and step-by-step solutions.

Your knowledge base contains:
- Detailed lessons for each SAT question type
- 50+ example questions with complete solutions
- Caddell Prep's specific teaching methodologies
- Common mistakes and how to avoid them

ALWAYS:
1. Search your knowledge base first for relevant lessons and examples
2. Use Caddell Prep's specific approaches and terminology
3. Provide step-by-step solutions matching the style in the materials
4. Reference similar examples from the knowledge base when helpful
5. Give the correct answer first, then detailed explanation
6. Use proper LaTeX formatting for math
7. Be encouraging and supportive of student learning

When students upload SAT questions or ask about topics, find the most relevant materials in your knowledge base and provide answers that align with Caddell Prep's teaching methods.""",
        "model": "gpt-4o",
        "tools": [
            {"type": "file_search"}
        ]
    }
    
    async with httpx.AsyncClient(timeout=30.0) as client:
        try:
            response = await client.post(
                "https://api.openai.com/v1/assistants",
                headers=headers,
                json=assistant_data
            )
            
            if response.status_code == 200:
                assistant = response.json()
                return {
                    "message": "Knowledge base assistant created successfully!",
                    "assistant_id": assistant["id"],
                    "instructions": f"Add this to your Railway environment variables: ASSISTANT_ID={assistant['id']}",
                    "next_steps": [
                        "1. Add ASSISTANT_ID environment variable in Railway",
                        "2. Upload your 10 PDFs to this assistant using OpenAI's interface",
                        "3. Test Bucky with questions from your materials"
                    ]
                }
            else:
                error_data = response.json()
                raise HTTPException(status_code=response.status_code, detail=f"Failed to create assistant: {error_data}")
                
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Setup failed: {str(e)}")

@app.post("/api/analyze-question", response_model=APIResponse)
async def analyze_question(file: UploadFile = File(...)):
    """Analyze uploaded SAT question using knowledge base"""
    
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if file.size and file.size > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Maximum size is 10MB")
    
    try:
        logger.info(f"Processing image with knowledge base: {file.filename}")
        
        # Read and encode the image
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # Prepare message for assistant with image
        message_content = [
            {
                "type": "text",
                "text": """Please analyze this math question image. Follow the Caddell Prep method:

1. Identify what type of question this is
2. Find similar examples in the knowledge base
3. Explain step-by-step using our teaching methods
4. Consider if Desmos calculator would help
5. Look for any Caddell Prep tricks (answer choices, plugging in numbers, etc.)

Provide the answer first, then detailed explanation matching our materials."""
            },
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}"
                }
            }
        ]
        
        response_content = await call_openai_assistants_api(message_content)
        
        logger.info("Successfully analyzed image using knowledge base")
        return APIResponse(response=response_content)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/api/chat", response_model=APIResponse)
async def chat(message_data: ChatMessage):
    """Handle text chat using knowledge base"""
    
    if not message_data.message or len(message_data.message.strip()) == 0:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if len(message_data.message) > 2000:
        raise HTTPException(status_code=400, detail="Message too long. Maximum 2000 characters")
    
    try:
        logger.info(f"Processing chat with knowledge base: {message_data.message[:50]}...")
        
        # Enhanced message for knowledge base search
        enhanced_message = f"""Student question: {message_data.message}

Please help using the Caddell Prep method:
1. Identify what type of math question this is
2. Search the knowledge base for similar examples
3. Explain step-by-step using our teaching style from the PDFs
4. Consider Desmos calculator if helpful: https://www.desmos.com/calculator
5. Look for Caddell Prep tricks mentioned in our materials
6. Be friendly and instructional like tutoring a confused student"""
        
        response_content = await call_openai_assistants_api(enhanced_message)
        
        logger.info("Successfully processed chat using knowledge base")
        return APIResponse(response=response_content)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
