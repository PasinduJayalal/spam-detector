# api/app.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow your frontend (we'll change this later if needed)
ALLOWED_ORIGINS = ["http://localhost:5173"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
def your_world():
    return {"status": "God is with you! Pasindu! Dont you dare give up!"}

@app.get("/health")
def health():
    # This is just a heartbeat. If you see {"status": "ok"}, the API runs!
    return {"status": "ok"}

@app.get("/your_world")
def your_world():
    return {"status": "The world is yours!"}