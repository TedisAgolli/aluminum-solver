"""
Local development server for the Aluminum Cutting Calculator.
Run this file to test locally before deploying to Vercel.
"""
import uvicorn
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# Import the routes from api/index.py
from api.index import solve, api_root, root

# Create main app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Add API routes  
app.get("/")(root)
app.post("/api/solve")(solve)
app.get("/api")(api_root)

# Mount static files last (catch-all)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    print("\n" + "="*60)
    print("🔧 Aluminum Cutting Calculator - Local Development Server")
    print("="*60)
    print("\n✅ Server starting at: http://localhost:8000")
    print("📋 API docs at: http://localhost:8000/api/docs")
    print("\n💡 Press Ctrl+C to stop the server\n")
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
