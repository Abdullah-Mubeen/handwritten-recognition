from fastapi import FastAPI
from app.config import settings


app = FastAPI(
    title=settings.project_name,
    version="1.0.0"
)

@app.get("/")
def read_root():
    return {"message": "Welcome"}
