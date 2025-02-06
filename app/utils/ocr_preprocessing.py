import cv2
import numpy as np
import logging

# Configure a logger for OCR utilities.
logger = logging.getLogger("ocr_utils")
logger.setLevel(logging.INFO)
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] in %(module)s: %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)

def load_image(image_path: str):
    """
    Load an image from the given path using OpenCV.
    """
    image = cv2.imread(image_path)
    if image is None:
        logger.error("Image loading failed. Check the file path.")
        raise ValueError("Could not load the image.")
    return image

def preprocess_image(image_path: str, save_debug: bool = True):
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not load the image.")

    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Resize the image to a higher resolution
    scale_percent = 150  # Increase size by 150%
    width = int(gray.shape[1] * scale_percent / 100)
    height = int(gray.shape[0] * scale_percent / 100)
    resized = cv2.resize(gray, (width, height), interpolation=cv2.INTER_CUBIC)
    
    # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) for better contrast
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
    enhanced = clahe.apply(resized)
    
    # Apply bilateral filter to reduce noise while preserving edges
    denoised = cv2.bilateralFilter(enhanced, 9, 75, 75)
    
    # Adaptive thresholding for better binarization
    thresh = cv2.adaptiveThreshold(
        denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 11, 2
    )
    
    # Morphological operations to enhance text visibility
    kernel = np.ones((2,2), np.uint8)
    morphed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    # Deskew the image
    deskewed = deskew_image(morphed)
    
    # Save preprocessed image for debugging
    if save_debug:
        debug_path = "preprocessed_image.png"
        cv2.imwrite(debug_path, deskewed)
        logger.info(f"Preprocessed image saved as {debug_path}")
    
    return deskewed

def deskew_image(image):
    """Deskew the image using Hough Line Transform."""
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=5)
    if lines is None:
        return image
    
    angles = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angles.append(angle)
    
    if len(angles) == 0:
        return image
    
    median_angle = np.median(angles)
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
    deskewed = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    
    return deskewed