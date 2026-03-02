# backend/app.py
import os
import cv2
import time
import base64
import requests
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime
from models import visits_collection

# Import YOUR existing YOLO+EasyOCR pipeline
from plate_ocr import extract_plate_from_base64

# Import DeepFace for Facial Recognition
try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    print("Warning: DeepFace is not installed. Face matching will be disabled.")

app = Flask(__name__)
CORS(app)

# ==========================================================
# Helper Function: Real-Time Face Match
# ==========================================================
def verify_face(cam_frame, stored_b64):
    """Compares the live webcam frame to the stored base64 photo"""
    if not DEEPFACE_AVAILABLE:
        return False
    try:
        encoded_data = stored_b64.split(',')[1] if ',' in stored_b64 else stored_b64
        nparr = np.frombuffer(base64.b64decode(encoded_data), np.uint8)
        stored_img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        result = DeepFace.verify(cam_frame, stored_img, enforce_detection=False)
        return result.get("verified", False)
    except Exception as e:
        print(f"Face verification error: {e}")
        return False

@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Live", "database": "AWS MongoDB Atlas Connected"}), 200

# ==========================================================
# 1. POST ROUTE (Create normal visit) - FIXED
# ==========================================================
# ==========================================================
# 1. POST ROUTE (Create normal visit) - UPDATED
# ==========================================================
@app.route("/api/visits", methods=["POST"])
def create_visit():
    try:
        data = request.json or {}
        
        # FIX: Safely handle the vehicle number. Frontend sends it as 'vehicleNo'
        # we check both potential keys to ensure data is captured correctly.
        raw_vehicle = data.get("vehicleNo") or data.get("vehicleNumber") or ""
        form_vehicle_number = str(raw_vehicle).upper().replace(" ", "")
        
        # Safely handle the photo base64 if it exists for OCR processing
        image_b64 = data.get("vehicleNoPhoto")
        
        ocr_plate = ""
        plate_match = False

        # Process the camera image through your YOLOv8+EasyOCR pipeline
        if image_b64:
            ocr_result = extract_plate_from_base64(image_b64)
            ocr_plate = ocr_result.get("plate", "").upper().replace(" ", "")
            
            # Match the detected text against the form number (substring match)
            if ocr_plate and form_vehicle_number:
                if form_vehicle_number in ocr_plate or ocr_plate in form_vehicle_number:
                    plate_match = True

        # Prepare metadata for storage
        data["ocr_plate_detected"] = ocr_plate
        data["plate_match_success"] = plate_match
        
        # Determine status: auto-approve if plate matches, otherwise mark for review
        data["status"] = "approved" if plate_match else "pending_review"
        data["submittedAt"] = datetime.utcnow()

        # Attempt to insert into AWS MongoDB
        try:
            result = visits_collection.insert_one(data)
        except Exception as db_err:
            # Handle unique index constraint for Student ID
            if "E11000" in str(db_err):
                return jsonify({"error": "A gate pass for this Student ID already exists."}), 400
            raise db_err

        return jsonify({
            "message": "Gate Pass Processed Successfully",
            "id": str(result.inserted_id),
            "detected_plate": ocr_plate,
            "match": plate_match
        }), 201

    except Exception as e:
        # Log exact error to terminal for debugging
        print(f"Backend Error in /api/visits: {e}") 
        return jsonify({"error": str(e)}), 500

@app.route("/api/visits", methods=["GET"])
def get_visits():
    try:
        cursor = visits_collection.find().sort("submittedAt", -1)
        visits = [{**doc, "_id": str(doc["_id"])} for doc in cursor]
        return jsonify(visits), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/visits/<receipt_id>/status", methods=["PUT"])
def update_visit_status(receipt_id):
    try:
        data = request.json
        new_status = data.get("status")
        result = visits_collection.update_one({"receiptId": receipt_id}, {"$set": {"status": new_status}})
        if result.matched_count == 0:
            return jsonify({"error": "Receipt not found"}), 404
        return jsonify({"message": f"Status updated to {new_status}"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/visits/<receipt_id>", methods=["DELETE"])
def delete_visit(receipt_id):
    try:
        result = visits_collection.delete_one({"receiptId": receipt_id})
        if result.deleted_count == 0:
            return jsonify({"error": "Receipt not found"}), 404
        return jsonify({"message": "Gate pass deleted successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/invite-admin", methods=["POST"])
def invite_admin():
    try:
        data = request.json
        email_to_invite = data.get("email")
        if not email_to_invite:
            return jsonify({"error": "Email is required"}), 400

        clerk_secret = os.getenv("CLERK_SECRET_KEY")
        if not clerk_secret:
            return jsonify({"error": "Clerk API key missing in backend"}), 500

        url = "https://api.clerk.com/v1/invitations"
        headers = {"Authorization": f"Bearer {clerk_secret}", "Content-Type": "application/json"}
        payload = {"email_address": email_to_invite, "public_metadata": {"role": "admin"}, "ignore_existing": True}

        response = requests.post(url, json=payload, headers=headers)
        if response.status_code in [200, 201]:
            return jsonify({"message": "Successfully sent Admin invite"}), 200
        return jsonify({"error": "Failed to send invite", "details": response.json()}), response.status_code
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# ==========================================================
# 6. POST ROUTE (Live Webcam Test: YOLOv8 + OCR + Face Match)
# ==========================================================
@app.route("/api/live-test", methods=["POST"])
def live_entry_test():
    try:
        # 1. Initialize OpenCV Webcam
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap = cv2.VideoCapture(0)
            
        if not cap.isOpened():
            return jsonify({"error": "Cannot access backend laptop webcam."}), 500

        time.sleep(1.5) # Allow camera sensor to warm up
        valid_frame = None
        
        for _ in range(15):
            ret, frame = cap.read()
            if ret and frame is not None:
                valid_frame = frame
                break 
            time.sleep(0.1)
            
        cap.release()
        cv2.destroyAllWindows()

        if valid_frame is None:
            return jsonify({"error": "Failed to capture image frame."}), 500

        # 2. Convert OpenCV frame to Base64
        _, buffer = cv2.imencode('.jpg', valid_frame)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        base64_string = f"data:image/jpeg;base64,{img_base64}"

        # 3. Detect License Plate using YOUR YOLOv8 logic!
        ocr_result = extract_plate_from_base64(base64_string)
        detected_plate = ocr_result.get("plate", "").upper().replace(" ", "")

        match_success = False
        face_match_success = False
        matched_form_data = None

        # 4. Search Database and Match Faces
        if detected_plate and len(detected_plate) > 3:
            all_visits = list(visits_collection.find())
            
            for visit in all_visits:
                db_plate = str(visit.get("vehicleNo", "")).upper().replace(" ", "")
                if db_plate and (db_plate in detected_plate or detected_plate in db_plate):
                    match_success = True
                    visit["_id"] = str(visit["_id"]) 
                    matched_form_data = visit
                    break

        # 5. If a matching car was found, verify the person's face!
        if match_success and matched_form_data and DEEPFACE_AVAILABLE:
            stored_photo = matched_form_data.get("receiptPhoto") or matched_form_data.get("photo")
            if not stored_photo and matched_form_data.get("members"):
                stored_photo = matched_form_data["members"][0].get("photo")

            if stored_photo:
                try:
                    face_match_success = verify_face(valid_frame, stored_photo)
                except Exception as face_err:
                    print(f"Face verification failed: {face_err}")

        return jsonify({
            "message": "Live Scan Executed Successfully",
            "detected_plate": detected_plate if detected_plate else "NO PLATE DETECTED",
            "match_success": match_success, 
            "plate_matched": match_success,
            "face_matched": face_match_success,
            "form_data": matched_form_data 
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True, use_reloader=False)