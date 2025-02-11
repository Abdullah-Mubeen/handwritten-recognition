from fastapi import FastAPI
from app.config import settings
from app.routes import ocr_upload


app = FastAPI(
    title=settings.project_name,
    version="1.0.0"
)

app.include_router(ocr_upload.router)


@app.get("/")
def read_root():
    return {"message": "Welcome"}
