import os
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import openai
import base64
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="SAT Tutor API", version="1.0.0")

# Allow your website to access the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # We'll make this more secure later
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Initialize OpenAI
client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

class ChatMessage(BaseModel):
    message: str

class APIResponse(BaseModel):
    response: str

# Your SAT tutor instructions (customize this!)
SAT_TUTOR_PROMPT = """You are an expert SAT tutor. When analyzing questions:
1. Give the correct answer first
2. Provide step-by-step explanations
3. Identify key concepts being tested
4. Mention common mistakes students make
Keep explanations clear and encouraging."""

@app.get("/")
async def root():
    return {"message": "SAT Tutor API is running successfully!"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/api/analyze-question")
async def analyze_question(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="Please upload an image file")
    
    try:
        # Read and encode the image
        contents = await file.read()
        base64_image = base64.b64encode(contents).decode('utf-8')
        
        # Send to OpenAI
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[{
                "role": "system",
                "content": SAT_TUTOR_PROMPT
            }, {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Analyze this SAT question image and provide the answer with explanation."
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }],
            max_tokens=1500
        )
        
        return {"response": response.choices[0].message.content}
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Sorry, there was an error analyzing the image")

@app.post("/api/chat")
async def chat(message_data: ChatMessage):
    if not message_data.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    
    try:
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": SAT_TUTOR_PROMPT},
                {"role": "user", "content": message_data.message}
            ],
            max_tokens=800
        )
        
        return {"response": response.choices[0].message.content}
        
    except Exception as e:
        logger.error(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail="Sorry, there was an error processing your message")

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
