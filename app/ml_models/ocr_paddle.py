from paddleocr import PaddleOCR
import cv2 
from app.utils.ocr_preprocessing import preprocess_image


# Load PaddleOCR model (English)
ocr_model = PaddleOCR(use_angle_cls=True, lang="en")

def recognize_text(image_path):
    """Runs OCR on a preprocessed image and returns extracted text."""
    # Preprocess the image
    processed_image = preprocess_image(image_path)

    # Perform OCR
    result = ocr_model.ocr(processed_image, cls=True)

    extracted_text = []
    for line in result:
        for word_info in line:
            extracted_text.append(word_info[1][0])  # Extract text

    return " ".join(extracted_text)

if __name__ == "__main__":
    image_path = r"E:\Projects\handwritten-recognition\app\data\Invoice1.png"  # Absolute path
    print("Extracted Text:", recognize_text(image_path))
