import cv2
import easyocr

# Initialize the EasyOCR reader
reader = easyocr.Reader(['en'])

def scan_vehicle_plate():
    """
    Accesses the local default camera, captures a frame, and performs OCR.
    """
    cap = cv2.VideoCapture(0) # 0 is the local camera index
    detected_no = ""
    
    if not cap.isOpened():
        return None, "Could not access local camera"

    ret, frame = cap.read()
    cap.release()

    if not ret:
        return None, "Failed to capture image"

    # Perform OCR on the frame
    results = reader.readtext(frame)
    
    for (bbox, text, prob) in results:
        # Basic alphanumeric filter for plates
        cleaned_text = text.upper().replace(" ", "")
        if len(cleaned_text) > 4:
            detected_no = cleaned_text
            break

    return detected_no, None