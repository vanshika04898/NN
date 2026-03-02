# backend/plate_ocr.py
import os
import cv2
import numpy as np
import base64
import easyocr
from ultralytics import YOLO

# Safely locate the model file in the current directory
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
YOLO_MODEL_PATH = os.path.join(BASE_DIR, "yolov8n.pt")

try:
    _plate_detector = YOLO(YOLO_MODEL_PATH)
except Exception as e:
    print(f"❌ Custom model failed: {e}. Falling back to default yolov8n.pt")
    _plate_detector = YOLO("yolov8n.pt")

# Initialize EasyOCR for English
_ocr_reader = easyocr.Reader(['en'], gpu=False)

def extract_plate_from_base64(image_b64):
    try:
        # Decode the Base64 image sent from the React local camera
        if "," in image_b64:
            image_b64 = image_b64.split(",")[1]
        img_bytes = base64.b64decode(image_b64)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Detect the license plate bounding box
        results = _plate_detector.predict(img, conf=0.25, verbose=False)
        plate_text = ""
        
        for r in results:
            for box in r.boxes:
                b = box.xyxy[0].cpu().numpy().astype(int)
                crop = img[b[1]:b[3], b[0]:b[2]]
                
                # Read text from the cropped plate
                if crop.size > 0:
                    ocr_res = _ocr_reader.readtext(crop)
                    if ocr_res:
                        plate_text = ocr_res[0][1].upper().replace(" ", "")
                        break
            if plate_text:
                break
                
        return {"plate": plate_text}
    except Exception as e:
        print(f"OCR Processing Error: {e}")
        return {"plate": None}