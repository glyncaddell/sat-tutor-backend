# Updated main.py for Railway with Knowledge Base Integration

import os
import httpx
import json
import logging
import asyncio
from fastapi import FastAPI, File, UploadFile, HTTPException, Request, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
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

@app.post("/api/analyze-question")
async def analyze_question(file: UploadFile = File(...), request: Request = None):
    """Analyze uploaded SAT question using knowledge base - Updated for Assistants API"""
    
    if not file.content_type or not file.content_type.startswith('image/'):
        error_msg = "File must be an image"
        if request and "multipart/form-data" in str(request.headers.get("content-type", "")):
            return HTMLResponse(f"<div style='padding: 20px; font-family: Arial;'><h3 style='color: #dc3545;'>Error</h3><p>{error_msg}</p></div>")
        raise HTTPException(status_code=400, detail=error_msg)
    
    if file.size and file.size > 10 * 1024 * 1024:
        error_msg = "File too large. Maximum size is 10MB"
        if request and "multipart/form-data" in str(request.headers.get("content-type", "")):
            return HTMLResponse(f"<div style='padding: 20px; font-family: Arial;'><h3 style='color: #dc3545;'>Error</h3><p>{error_msg}</p></div>")
        raise HTTPException(status_code=400, detail=error_msg)
    
    try:
        logger.info(f"Processing image with knowledge base: {file.filename}")
        
        # Read and encode the image
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # For Assistants API, we need to use the direct HTTP approach for vision
        response_content = await call_openai_vision_with_knowledge_base(base64_image)
        
        logger.info("Successfully analyzed image using knowledge base")
        
        # Return HTML for form submissions, JSON for API calls
        if request and "multipart/form-data" in str(request.headers.get("content-type", "")):
            html_response = f"""
            <div style='padding: 20px; font-family: Arial, sans-serif; line-height: 1.6; max-width: 100%;'>
                <div style='background: #e8f5e8; border-left: 4px solid #28a745; padding: 15px; margin-bottom: 15px; border-radius: 0 6px 6px 0;'>
                    <strong style='color: #155724;'>🎓 Bucky's Image Analysis:</strong>
                </div>
                <div style='background: white; padding: 15px; border-radius: 6px; border: 1px solid #ddd; white-space: pre-wrap;'>
                    {response_content.replace('<', '&lt;').replace('>', '&gt;')}
                </div>
                <div style='margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 6px; font-size: 12px; color: #6c757d;'>
                    Analyzed using Caddell Prep's teaching methods
                </div>
            </div>
            """
            return HTMLResponse(html_response)
        else:
            return APIResponse(response=response_content)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error analyzing image: {str(e)}")
        if request and "multipart/form-data" in str(request.headers.get("content-type", "")):
            return HTMLResponse(f"<div style='padding: 20px; font-family: Arial;'><h3 style='color: #dc3545;'>Error</h3><p>Internal server error: {str(e)}</p></div>")
        raise HTTPException(status_code=500, detail="Internal server error")

async def call_openai_vision_with_knowledge_base(base64_image):
    """Call OpenAI Vision API with knowledge base context"""
    
    if not OPENAI_API_KEY:
        raise HTTPException(status_code=500, detail="OpenAI API key not configured")
    
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json"
    }
    
    # Enhanced prompt that incorporates your knowledge base approach
    system_prompt = """You are Bucky, a math tutor working for Caddell Prep. Analyze this math question image using our proven teaching methods:

1. IDENTIFY what type of question this is (e.g., system of equations, quadratic, function graph)
2. EXPLAIN the approach step-by-step using Caddell Prep methods
3. PROVIDE the correct answer first, then detailed explanation
4. CONSIDER if Desmos calculator would help: https://www.desmos.com/calculator
5. LOOK for Caddell Prep tricks: answer choice testing, plugging in numbers, graphing
6. USE proper math formatting: $x^2$ for inline, $equations$ for display
7. BE friendly and instructional, like tutoring a confused student
8. AVOID shortcuts unless they're part of the Caddell Prep method

Based on your training with Caddell Prep materials, use the same step-by-step approach and teaching style from the uploaded lesson PDFs."""

    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Please analyze this math question image using the Caddell Prep method. Identify the question type, provide the answer, and explain step-by-step using our teaching approach."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1500,
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

@app.post("/api/chat")
async def chat(message_data: ChatMessage = None, request: Request = None):
    """Handle text chat using knowledge base - supports both JSON and form data"""
    
    message_text = None
    
    # Try to get message from JSON first
    if message_data and message_data.message:
        message_text = message_data.message
    else:
        # Try to get from form data
        try:
            from fastapi import Request, Form
            form = await request.form()
            message_text = form.get("message")
        except:
            pass
    
    if not message_text or len(message_text.strip()) == 0:
        # Return HTML error for form submissions, JSON for API calls
        if request and request.headers.get("content-type") == "application/x-www-form-urlencoded":
            return HTMLResponse("<div style='padding: 20px; font-family: Arial;'><h3 style='color: #dc3545;'>Error</h3><p>Message cannot be empty. Please enter your question.</p></div>")
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    if len(message_text) > 2000:
        if request and request.headers.get("content-type") == "application/x-www-form-urlencoded":
            return HTMLResponse("<div style='padding: 20px; font-family: Arial;'><h3 style='color: #dc3545;'>Error</h3><p>Message too long. Maximum 2000 characters.</p></div>")
        raise HTTPException(status_code=400, detail="Message too long. Maximum 2000 characters")
    
    try:
        logger.info(f"Processing chat with knowledge base: {message_text[:50]}...")
        
        enhanced_message = f"""Student question: {message_text}

Please help using the Caddell Prep method:
1. Identify what type of math question this is
2. Search the knowledge base for similar examples
3. Explain step-by-step using our teaching style from the PDFs
4. Consider Desmos calculator if helpful: https://www.desmos.com/calculator
5. Look for Caddell Prep tricks mentioned in our materials
6. Be friendly and instructional like tutoring a confused student"""
        
        response_content = await call_openai_assistants_api(enhanced_message)
        
        logger.info("Successfully processed chat using knowledge base")
        
        # Return HTML for form submissions, JSON for API calls
        if request and "application/x-www-form-urlencoded" in str(request.headers.get("content-type", "")):
            # Return HTML response for form submission
            html_response = f"""
            <div style='padding: 20px; font-family: Arial, sans-serif; line-height: 1.6; max-width: 100%;'>
                <div style='background: #e8f5e8; border-left: 4px solid #28a745; padding: 15px; margin-bottom: 15px; border-radius: 0 6px 6px 0;'>
                    <strong style='color: #155724;'>🎓 Bucky's Response:</strong>
                </div>
                <div style='background: white; padding: 15px; border-radius: 6px; border: 1px solid #ddd; white-space: pre-wrap;'>
                    {response_content.replace('<', '&lt;').replace('>', '&gt;')}
                </div>
                <div style='margin-top: 15px; padding: 10px; background: #f8f9fa; border-radius: 6px; font-size: 12px; color: #6c757d;'>
                    Powered by Caddell Prep's proven teaching methods
                </div>
            </div>
            """
            return HTMLResponse(html_response)
        else:
            return APIResponse(response=response_content)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing chat: {str(e)}")
        if request and "application/x-www-form-urlencoded" in str(request.headers.get("content-type", "")):
            return HTMLResponse(f"<div style='padding: 20px; font-family: Arial;'><h3 style='color: #dc3545;'>Error</h3><p>Sorry, there was an error: {str(e)}</p></div>")
        raise HTTPException(status_code=500, detail="Internal server error")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
