import os
import shutil
import logging
from fastapi import APIRouter, File, UploadFile, HTTPException, status
from fastapi.responses import JSONResponse
from paddleocr import PaddleOCR

# Configure logger
logger = logging.getLogger("ocr_upload")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] in %(module)s: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

# Initialize FastAPI Router
router = APIRouter(
    prefix="/ocr",
    tags=["OCR"],
    responses={404: {"description": "Not found"}}
)

# Upload Directory
UPLOAD_DIRECTORY = os.path.join(os.getcwd(), "uploads")
os.makedirs(UPLOAD_DIRECTORY, exist_ok=True)

# Initialize PaddleOCR model
ocr_model = PaddleOCR(use_angle_cls=True, lang="en")

def is_valid_image(filename: str) -> bool:
    """Check if the uploaded file is a valid image format."""
    valid_extensions = (".png", ".jpg", ".jpeg", ".bmp", ".tiff")
    return filename.lower().endswith(valid_extensions)

@router.post("/upload", status_code=status.HTTP_201_CREATED)
async def upload_ocr_image(file: UploadFile = File(...)):
    """Upload an image and perform OCR using PaddleOCR."""
    logger.info("Received OCR file upload request.")

    if not is_valid_image(file.filename):
        logger.error(f"Invalid file type: {file.filename}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Supported types: PNG, JPG, JPEG, BMP, TIFF."
        )
    
    try:
        # Save the uploaded file
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

    # Perform OCR
    try:
        logger.info("Performing OCR...")
        result = ocr_model.ocr(file_path, cls=True)

        extracted_text = []
        for line in result:
            for word_info in line:
                extracted_text.append(word_info[1][0])  # Extract detected text

        final_text = " ".join(extracted_text)
        logger.info("OCR processing completed successfully.")

    except Exception as error:
        logger.exception("OCR processing failed.")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="An error occurred during OCR processing."
        ) from error

    return JSONResponse(
        status_code=status.HTTP_201_CREATED,
        content={
            "message": "OCR completed successfully",
            "extracted_text": final_text,
            "file_path": file_path
        }
    )
