import os
import shutil
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from app.utils import ocr_preprocessing  # Import the OCR utilities

# Configure logger for this module
logger = logging.getLogger("ocr_upload")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] in %(module)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

router = APIRouter(
    prefix="/ocr",
    tags=["OCR"],
    responses={404: {"description": "Not found"}}
)

UPLOAD_DIRECTORY = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

def is_valid_image(filename: str) -> bool:
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    return filename.lower().endswith(valid_extensions)

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_ocr_image(file: UploadFile = File(...)):
    logger.info("Received file upload request for OCR processing.")

    if not is_valid_image(file.filename):
        logger.error(f"Invalid file type: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Supported types: PNG, JPG, JPEG, BMP, TIFF."
        )
    
    try:
        safe_filename = os.path.basename(file.filename)
        file_path = os.path.join(UPLOAD_DIRECTORY, safe_filename)
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        logger.info(f"File saved successfully at: {file_path}")
    
    except Exception as error:
        logger.exception("Failed to save the uploaded file.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred while saving the file."
        ) from error

    # Preprocess the image
    try:
        processed_image = ocr_preprocessing.preprocess_image(file_path)
        logger.info("Image preprocessed successfully.")
    except Exception as error:
        logger.exception("Image preprocessing failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during image preprocessing."
        ) from error

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "message": "File uploaded and preprocessed successfully",
            "file_path": file_path
        }
    )
