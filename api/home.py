from fastapi import FastAPI
from fastapi.responses import HTMLResponse
import os

app = FastAPI()

@app.get("/")
async def home():
    # Read the HTML file
    html_path = os.path.join(os.path.dirname(__file__), "..", "public", "index.html")
    with open(html_path, "r") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)
