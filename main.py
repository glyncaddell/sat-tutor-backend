import os
from fastapi import FastAPI

app = FastAPI(title="SAT Tutor API", version="1.0.0")

@app.get("/")
async def root():
    return {"message": "SAT Tutor API is running successfully!", "status": "healthy"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
